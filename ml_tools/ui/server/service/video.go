package service

// Keep in sync with ml_tools/ui/service/video.py

import (
	"encoding/binary"
	"fmt"
	"ml_tools/server/protocol"
	"sync"
)

type StreamUpdateChan chan []byte
type Video struct {
	pw protocol.ReadWriter

	streamUpdateChanMap sync.Map
}

func newVideo(pw protocol.ReadWriter) *Video {
	return &Video{
		pw: pw,

		streamUpdateChanMap: sync.Map{},
	}
}

func (v *Video) GetMessageTypeRange() (min, max int) {
	return 200, 399
}

func (v *Video) OnMessage(message protocol.Message) {
	switch message.Type {
	case UpdateStream:
		v.handleUpdateStream(message)
	}
}

func (v *Video) handleUpdateStream(message protocol.Message) {
	d := message.Data

	nameLen := binary.LittleEndian.Uint32(d[:4])
	d = d[4:]

	name := string(d[:nameLen])
	d = d[nameLen:]

	streamUpdateBytes := d

	streamUpdateChanI, ok := v.streamUpdateChanMap.Load(name)
	var streamUpdateChan StreamUpdateChan
	if !ok {
		streamUpdateChan = make(StreamUpdateChan)
		streamUpdateChanI, loaded := v.streamUpdateChanMap.LoadOrStore(name, streamUpdateChan)
		if loaded {
			streamUpdateChan = streamUpdateChanI.(StreamUpdateChan)
		}
	} else {
		streamUpdateChan = streamUpdateChanI.(StreamUpdateChan)
	}

	select {
	case streamUpdateChan <- streamUpdateBytes:
	default:
	}
}

func (v *Video) GetStreamUpdateChan(streamName string) (StreamUpdateChan, error) {
	streamUpdateChanI, ok := v.streamUpdateChanMap.Load(streamName)
	if !ok {
		return nil, fmt.Errorf("Error: Stream with name: %s does not exist", streamName)
	}

	return streamUpdateChanI.(StreamUpdateChan), nil
}
