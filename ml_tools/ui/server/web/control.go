package web

import (
	"fmt"
	"io"
	"log"
	"net/http"
)

func (w *Web) startControl() {
	w.serverMux.HandleFunc("/mlTools/control/set_value", func(rw http.ResponseWriter, r *http.Request) {
		const logPrefix = "Control set_value handler:"
		if r.Method != http.MethodPost {
			return
		}

		var err error
		defer func() {
			if err != nil {
				rw.WriteHeader(http.StatusInternalServerError)
			}
		}()

		valueBytes, err := io.ReadAll(r.Body)
		if err != nil {
			log.Println(logPrefix, "io.ReadAll valueBytes:", err)
			return
		}

		keySlice, ok := r.URL.Query()["key"]
		if !ok {
			err = fmt.Errorf("no 'key' in URL Query")
			log.Println(logPrefix, "r.URL.Query:", err)
			return
		}
		key := keySlice[0]

		value := string(valueBytes)
		err = w.services.Control.SetValue(key, value)
		if err != nil {
			log.Println(logPrefix, "w.services.Control.SetValue:", err)
			return
		}

		rw.WriteHeader(http.StatusOK)
	})
}
