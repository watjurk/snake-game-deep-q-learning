package web

import (
	"errors"
	"fmt"
	"ml_tools/server/service"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"syscall"
)

type Web struct {
	publicPath string

	serverMux *http.ServeMux
	services  *service.Services
}

func New(s *service.Services, publicPath string) *Web {
	w := &Web{
		publicPath: publicPath,

		serverMux: http.NewServeMux(),
		services:  s,
	}

	return w
}

func (w *Web) Start() {
	w.startControl()
	w.startVideo()

	// This path must be relative to root of the server directory.
	mlToolsPublic, err := filepath.Abs("./web/mlTools")
	if err != nil {
		panic(err)
	}

	// Register our internal js helper files.
	mlToolsPattern := "/mlTools/"
	w.serverMux.Handle(mlToolsPattern, http.StripPrefix(mlToolsPattern, http.FileServer(http.Dir(mlToolsPublic))))

	// Register user public folder.
	w.serverMux.Handle("/", http.FileServer(http.Dir(w.publicPath)))

	go func() {
		port := 8080
		var ln net.Listener

		for {
			ln, err = net.Listen("tcp", fmt.Sprintf(":%v", port))
			if err != nil {
				if isErrorAddressAlreadyInUse(err) {
					port++
					continue
				}

				panic(err)
			}

			break
		}

		addr := ln.Addr().String()
		fmt.Printf("Server on: %v\n", addr)

		server := http.Server{Addr: addr, Handler: w.serverMux}
		err = server.Serve(ln)
		if err != nil {
			panic(err)
		}
	}()
}

func isErrorAddressAlreadyInUse(err error) bool {
	var eOsSyscall *os.SyscallError
	if !errors.As(err, &eOsSyscall) {
		return false
	}
	var errErrno syscall.Errno // doesn't need a "*" (ptr) because it's already a ptr (uintptr)
	if !errors.As(eOsSyscall, &errErrno) {
		return false
	}
	if errErrno == syscall.EADDRINUSE {
		return true
	}
	const WSAEADDRINUSE = 10048
	if runtime.GOOS == "windows" && errErrno == WSAEADDRINUSE {
		return true
	}
	return false
}
