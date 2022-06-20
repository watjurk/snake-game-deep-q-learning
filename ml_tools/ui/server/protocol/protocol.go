package protocol

import (
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"io"
)

type delimerT [16]byte

var NilMessage Message

type MessageType int32

type Message struct {
	Type MessageType
	Data []byte
}

func messageFromBytes(m *Message, messageBytes []byte) {
	m.Type = MessageType(binary.LittleEndian.Uint32(messageBytes[:4]))

	messageData := messageBytes[4:]
	m.Data = make([]byte, len(messageData))
	copy(m.Data, messageData)
}

func messageToBytes(m Message, w io.Writer) error {
	messageType := [4]byte{}
	binary.LittleEndian.PutUint32(messageType[:], uint32(m.Type))

	_, err := w.Write(messageType[:])
	if err != nil {
		return err
	}

	_, err = w.Write(m.Data)
	return err
}

type protocol struct {
	decoder
	encoder
	delimer delimerT
}

type ReadWriter interface {
	reader
	writer
}

type reader interface {
	Read() (Message, error)
}

type writer interface {
	Write(m Message) error
}

func NewProtocol(rw io.ReadWriter) (ReadWriter, error) {
	var delimer delimerT
	_, err := rand.Read(delimer[:])
	if err != nil {
		return nil, err
	}

	_, err = rw.Write(delimer[:])
	if err != nil {
		return nil, fmt.Errorf("Error while writing delimer: %e", err)
	}

	p := &protocol{
		delimer: delimer,
		decoder: newDecoder(rw, delimer),
		encoder: newEncoder(rw, delimer),
	}

	return p, nil
}

type decoder struct {
	r       io.Reader
	delimer delimerT

	needRead   bool
	buffer     []byte
	readBuffer []byte
}

func newDecoder(r io.Reader, delimer delimerT) decoder {
	return decoder{
		r:       r,
		delimer: delimer,

		needRead:   true,
		buffer:     make([]byte, 0, 1024),
		readBuffer: make([]byte, 128),
	}
}

func (d *decoder) Read() (Message, error) {
	offset := 0
	for {
		if d.needRead {
			n, err := d.r.Read(d.readBuffer)
			if err != nil {
				return NilMessage, err
			}

			d.buffer = append(d.buffer, d.readBuffer[:n]...)
			d.needRead = false
		}

		// delimerSafeOffset is required when delimer is split between reads.
		delimerSafeOffset := offset - len(d.delimer)
		isUsingDelimerSafeOffset := true
		if delimerSafeOffset < 0 {
			delimerSafeOffset = offset
			isUsingDelimerSafeOffset = false
		}

		searchBytes := d.buffer[delimerSafeOffset:]
		searchBytesIndex := 0

		delimerIndex := 0
		delimerFound := false

		for searchBytesIndex = 0; searchBytesIndex < len(searchBytes); searchBytesIndex++ {
			if searchBytes[searchBytesIndex] == d.delimer[delimerIndex] {
				delimerIndex++
			} else {
				delimerIndex = 0
			}

			if delimerIndex == len(d.delimer) {
				delimerFound = true
				break
			}
		}

		offset += searchBytesIndex
		if isUsingDelimerSafeOffset {
			offset -= len(d.delimer)
		}

		if delimerFound {
			messageBytes := d.buffer[:offset-len(d.delimer)+1]
			message := Message{}
			messageFromBytes(&message, messageBytes)

			newBuffer := d.buffer[offset+1:]
			copy(d.buffer, newBuffer)
			d.buffer = d.buffer[:len(newBuffer)]

			return message, nil
		}

		d.needRead = true
	}
}

type encoder struct {
	w       io.Writer
	delimer delimerT
}

func newEncoder(w io.Writer, delimer delimerT) encoder {
	return encoder{
		w:       w,
		delimer: delimer,
	}
}

func (e *encoder) Write(m Message) error {
	err := messageToBytes(m, e.w)
	if err != nil {
		return err
	}

	_, err = e.w.Write(e.delimer[:])
	return err
}
