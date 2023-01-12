package service

// Keep in sync with ml_tools/ui/service/control.py

import (
	"encoding/json"
	"ml_tools/server/protocol"
)

type Control struct {
	pw protocol.ReadWriter
}

func (c *Control) GetMessageTypeRange() (min, max int) {
	return 100, 199
}

type setValueModel struct {
	Key   string `json:"key"`
	Value string `json:"value"`
}

func (c *Control) SetValue(key, value string) error {
	data, err := json.Marshal(setValueModel{
		Key:   key,
		Value: value,
	})
	if err != nil {
		return nil
	}

	return c.pw.Write(protocol.Message{
		Type: SetValue,
		Data: data,
	})
}

func (c *Control) OnMessage(message protocol.Message) {}
