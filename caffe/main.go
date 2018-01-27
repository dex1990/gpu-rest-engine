package main

// #cgo pkg-config: opencv cudart-8.0
// #cgo LDFLAGS: -lprocess -L. -L/root/Downloads/caffe/build/lib -lcaffe -L/usr/lib64 -lglog -lboost_system -lboost_thread
// #cgo CXXFLAGS: -std=c++11 -I/root/Downloads/caffe/include -I.. -O2 -fomit-frame-pointer -Wall
// #include <stdlib.h>
// #include "classification.h"
import "C"
import "unsafe"

import (
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
)


var ctxlist *C.classifier_ctxlist

func classify(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "", http.StatusMethodNotAllowed)
		return
	}
	buffer, err := ioutil.ReadAll(r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	
        }
	cstr, err := C.classifier_classify(ctxlist, (*C.char)(unsafe.Pointer(&buffer[0])))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	defer C.free(unsafe.Pointer(cstr))
	io.WriteString(w, C.GoString(cstr))
}

func main() {
	cmodel := C.CString(os.Args[1])
	ctrained := C.CString(os.Args[2])
	cmean := C.CString(os.Args[3])
	clabel := C.CString(os.Args[4])
	log.Println("Initializing Caffe classifiers")
	var err error
	ctxlist, err = C.classifier_initialize(cmodel, ctrained, cmean, clabel)
	if err != nil {
		log.Fatalln("could not initialize classifier:", err)
		return
	}
	defer C.classifier_destroy(ctxlist)

	log.Println("Adding REST endpoint /api/classify")
	http.HandleFunc("/api/classify", classify)
	log.Println("Starting server listening on " + os.Args[5])
	log.Fatal(http.ListenAndServe(os.Args[5], nil))
}
