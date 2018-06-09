package main

// #cgo pkg-config: cudart-8.0
// #cgo LDFLAGS: -lglog -lopencv_core -lopencv_imgcodecs -lopencv_cudawarping -lopencv_cudaarithm -lopencv_cudaimgproc -L/DIT/trunk/build/linux -lnvinfer -lnvcaffe_parser
// #cgo CXXFLAGS: -std=c++11 -I"/DIT/trunk/engine" -I"/DIT/trunk/caffeParser" -O2 -fomit-frame-pointer -Wall
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
	"strconv"
	"time"
)

var ctx *C.classifier_ctx

func GIEInference_Handler(w http.ResponseWriter, r *http.Request) {

	cstr := GIEInference(w, r)
	io.WriteString(w, cstr)
}

func GIEInference_Performance_Handler(w http.ResponseWriter, r *http.Request) {

	start := time.Now()
	GIEInference(w, r)
	elapsed := time.Since(start).Seconds()
	time_in_string := strconv.FormatFloat(elapsed, 'f', 6, 64)
	io.WriteString(w, time_in_string)
}

func GIEInference(w http.ResponseWriter, r *http.Request) string {
	if r.Method != "POST" {
		http.Error(w, "", http.StatusMethodNotAllowed)
		return ""
	}

	buffer, err := ioutil.ReadAll(r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return err.Error()
	}

	cstr, err := C.classifier_classify(ctx, (*C.char)(unsafe.Pointer(&buffer[0])), C.size_t(len(buffer)))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return err.Error()
	}

	defer C.free(unsafe.Pointer(cstr))
	return C.GoString(cstr)
}

func main() {
	cplan := C.CString(os.Args[1])
	cmean := C.CString(os.Args[2])
	clabel := C.CString(os.Args[3])

        //cmodel := C.CString(os.Args[1])
        //ctrained := C.CString(os.Args[2])
        //cmean := C.CString(os.Args[3])
        //clabel := C.CString(os.Args[4])

	log.Println("Initializing GIE classifiers")
	var err error
	ctx, err = C.classifier_init_with_plan(cplan, cmean, clabel)
	//ctx, err = C.classifier_init_with_model(cmodel, ctrained, cmean, clabel)
	if err != nil {
		log.Fatalln("could not initialize classifier:", err)
		return
	}
	defer C.classifier_destroy(ctx)

	log.Println("Adding REST endpoint /GIEInference/")
	http.HandleFunc("/GIEInference/", GIEInference_Handler)
	http.HandleFunc("/GIEInference_performance/", GIEInference_Performance_Handler)
	log.Println("Starting server listening on :8000")
	log.Fatal(http.ListenAndServe(":8000", nil))
}
