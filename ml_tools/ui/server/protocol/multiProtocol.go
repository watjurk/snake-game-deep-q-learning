package protocol

import (
	"io"
	"runtime"
	"sync"
)

type messageErr struct {
	message Message
	err     error
}

type MultiProtcol struct {
	writersMu sync.RWMutex
	writers   []writer

	messageChan chan messageErr
}

func NewMultiProtcol() *MultiProtcol {
	return &MultiProtcol{
		writersMu: sync.RWMutex{},

		// The runtime.NumCPU() is used because python will connect to us at least this amount of times.
		writers: make([]writer, 0, runtime.NumCPU()),

		// The 50 message buffer is arbitrary chosen value.
		messageChan: make(chan messageErr, 50),
	}
}

func (mp *MultiProtcol) AddProtcol(p ReadWriter) {
	mp.addProtcol(p)
	go mp.startReadingloop(p)
}

func (mp *MultiProtcol) AddProtcolBlocking(p ReadWriter) {
	mp.addProtcol(p)
	mp.startReadingloop(p)
}

func (mp *MultiProtcol) addProtcol(p ReadWriter) {
	mp.writersMu.Lock()
	mp.writers = append(mp.writers, p)
	mp.writersMu.Unlock()
}

func (mp *MultiProtcol) startReadingloop(r reader) {
	for {
		message, err := r.Read()
		mp.messageChan <- messageErr{message, err}
		if err == io.EOF {
			return
		}
	}
}

func (mp *MultiProtcol) Write(m Message) error {
	mp.writersMu.RLock()
	for _, writer := range mp.writers {
		err := writer.Write(m)
		if err != nil {
			return err
		}
	}
	mp.writersMu.RUnlock()
	return nil
}

func (mp *MultiProtcol) Read() (Message, error) {
	messageError := <-mp.messageChan
	return messageError.message, messageError.err
}
