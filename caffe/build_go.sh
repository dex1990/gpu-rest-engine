nvcc  -c process.cu  -o libprocess.a -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61
rm -rf /usr/local/go/bin/caffe-server
go get -ldflags="-s -w" caffe-server
caffe-server /root/Downloads/models/prototxt_list.txt /root/Downloads/models/caffemodel_list.txt imagenet_mean.binaryproto /root/Downloads/models/qnet_labels.txt :8000
