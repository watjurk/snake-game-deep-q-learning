package web

import (
	"errors"
	"fmt"
	"log"
	"mime/multipart"
	"net/http"
	"net/textproto"
	"syscall"
)

func (w *Web) startVideo() {
	w.serverMux.HandleFunc("/mlTools/video/stream", func(rw http.ResponseWriter, r *http.Request) {
		const logPrefix = "Video stream handler:"
		if r.Method != http.MethodGet {
			return
		}

		var err error
		defer func() {
			if err != nil {
				rw.WriteHeader(http.StatusInternalServerError)
			}
		}()

		streamNameSlice, ok := r.URL.Query()["name"]
		if !ok {
			err = fmt.Errorf("no 'name' in URL Query")
			log.Println(logPrefix, "r.URL.Query:", err)
			return
		}
		streamName := streamNameSlice[0]

		streamUpdateChan, err := w.services.Video.GetStreamUpdateChan(streamName)
		if err != nil {
			log.Println(logPrefix, "w.services.Video.GetStreamUpdateChan", err)
			return
		}

		multipartWriter := multipart.NewWriter(rw)
		rw.Header().Add("Content-Type", fmt.Sprintf(`multipart/x-mixed-replace; boundary="%s"`, multipartWriter.Boundary()))

		requestContext := r.Context()
		flusher, ok := rw.(http.Flusher)
		if !ok {
			flusher = noopFlusher{}
		}

		for streamUpdate := range streamUpdateChan {
			select {
			case <-requestContext.Done():
				break
			default:
			}

			partHeader := textproto.MIMEHeader{}
			partHeader.Set("Content-Type", "image/jpeg")
			part, err := multipartWriter.CreatePart(partHeader)
			if err != nil {
				if errors.Is(err, syscall.EPIPE) {
					return
				}
				log.Println(logPrefix, "multipartWriter.CreatePart:", err)
				continue
			}

			_, err = part.Write(streamUpdate)
			if err != nil {
				if errors.Is(err, syscall.EPIPE) {
					return
				}
				log.Println(logPrefix, "part.Write:", err)
				continue
			}
			flusher.Flush()
		}
	})
}

type noopFlusher struct{}

func (noopFlusher) Flush() {}
