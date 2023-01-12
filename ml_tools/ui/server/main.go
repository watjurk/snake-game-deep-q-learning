package main

import (
	"flag"
	"fmt"
	"log"
	"ml_tools/server/protocol"
	"ml_tools/server/web"
	"net"
	"os"
	"os/signal"
	"sync/atomic"

	"ml_tools/server/service"
)

func main() {
	publicPath := flag.String("public_folder_path", "", "")
	flag.Parse()

	if *publicPath == "" {
		log.Fatalln("Error: public_folder_path must be set.")
	}

	// Ignore interrupt because signal from jupyter notebook
	// will be propagated and server would die.
	signal.Ignore(os.Interrupt)

	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		panic(err)
	}
	defer l.Close()

	// Signal to python process our port.
	fmt.Println(l.Addr())

	// All connections are connected with multiProtcol so that we can send
	// everything and receive evertything in one place.
	mp := protocol.NewMultiProtcol()
	services := service.NewServies(mp)
	services.Start()

	w := web.New(services, *publicPath)
	w.Start()

	var connNumber int32
	for {
		conn, err := l.Accept()
		if err != nil {
			panic(err)
		}

		connNumber++
		go func() {
			serveConn(conn, mp)
			if atomic.AddInt32(&connNumber, -1) == 0 {
				// Exit when no one is connected to us.
				os.Exit(0)
			}
		}()
	}
}

func serveConn(conn net.Conn, mp *protocol.MultiProtcol) {
	defer conn.Close()

	proto, err := protocol.NewProtocol(conn)
	if err != nil {
		panic(err)
	}

	mp.AddProtcolBlocking(proto)
}
