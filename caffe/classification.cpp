#include "classification.h"
#include <iosfwd>
#include <vector>
#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <math.h>
#define USE_CUDNN 1
#include <caffe/caffe.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cuda_runtime.h>
#include "common.h"
#include "gpu_allocator.h"

using namespace caffe;
using std::string;
using GpuMat = cv::cuda::GpuMat;
using namespace cv;
/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

/* Based on the cpp_classification example of Caffe, but with GPU
 * image preprocessing and a simple memory pool. */
class Classifier
{
public:
    Classifier(const string& model_file,
               const string& trained_file,
               const string& mean_file,
               const string& label_file);

    std::vector<std::vector<Prediction> > Classify(const Mat& img, int N = 5);
    
    ~Classifier()
    {
       cudaStreamDestroy(stream_);
    }

    cudaStream_t CUDAStream()
    {
        return stream_;
    }

private:

    std::vector<std::vector<float> > PredictCuda(const Mat& img);    

    void WarmUp();

    void Preprocess(const Mat& img,std::vector<GpuMat>* input_channels);
   
private:
    std::shared_ptr<Net<float>> net_;
    Size input_geometry_;
    int num_channels_;
    GpuMat mean_;
    std::vector<string> labels_;
    cudaStream_t stream_;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file)
{
    Caffe::set_mode(Caffe::GPU);

    /* Load the network. */
    net_ = std::make_shared<Net<float>>(model_file, TEST);
    net_->CopyTrainedLayersFrom(trained_file);

    CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
    CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK(num_channels_ == 3 || num_channels_ == 1)
        << "Input layer should have 1 or 3 channels.";
    input_geometry_ = Size(input_layer->width(), input_layer->height());

    /* Load labels. */
    std::ifstream labels(label_file.c_str());
    CHECK(labels) << "Unable to open labels file " << label_file;
    string line;
    while (std::getline(labels, line))
        labels_.push_back(string(line));
    
    Blob<float>* output_layer = net_->output_blobs()[0];
    CHECK_EQ(labels_.size(), output_layer->channels())
        << "Number of labels is different from the output layer dimension.";
    cudaStreamCreate(&stream_);
//    printf("stream in Classifier %d\n",stream_);
    WarmUp();
    
}

extern "C" int Cudacvt(float *c,unsigned char *a, int w, int h, int tile_w, int tile_h, int idx,cudaStream_t stream);
/* Return the top N predictions. */
std::vector<std::vector<Prediction> > Classifier::Classify(const Mat& img, int N)
{   
    std::vector<std::vector<float> > outputs = PredictCuda(img);
    N = std::min<int>(labels_.size(), N);
    std::vector<std::vector<Prediction> > all_predictions;
    unsigned int j;
    int i;
    for ( j = 0; j < outputs.size(); ++j) {
     std::vector<float> output = outputs[j];
     N = std::min<int>(labels_.size(), N);
     std::vector<Prediction> predictions;
     for ( i = 0; i < N; ++i) {
       predictions.push_back(std::make_pair(labels_[i], output[i]));
     }
     all_predictions.push_back(predictions);
   }
   return all_predictions;   
}

std::vector<std::vector<float> > Classifier::PredictCuda(const Mat& img)
{
    Mat img_crop;
    int h = img.rows;
    int w = img.cols;
    int tile_w,tile_h;
    if(w < input_geometry_.width || h < input_geometry_.height)
    {    
        cv::resize(img,img_crop,input_geometry_);
        tile_w = 1;
        tile_h = 1;
    }
    else
    {
        int num_tile_w = cvFloor((double)w/input_geometry_.width);
        tile_w = (num_tile_w > 8) ? 8 : num_tile_w;
        int num_tile_h = cvFloor((double)h/input_geometry_.height);
        tile_h = (num_tile_h > 4) ? 4 : num_tile_h;
        img(Rect(0, 0, input_geometry_.width*tile_w,input_geometry_.height*tile_h)).copyTo(img_crop);
    }
        
    Blob<float>* input_layer = net_->input_blobs()[0];
    input_layer->Reshape(tile_w*tile_h, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    // Forward dimension change to all layers.
    net_->Reshape();

    cudaStream_t stream;
    stream = CUDAStream();
    Cudacvt(input_layer->mutable_gpu_data(),img_crop.data,input_geometry_.width,input_geometry_.height,tile_w,tile_h,0,stream);
    
    net_->Forward();
    /* Copy the output layer to a std::vector */
    std::vector<std::vector<float> > outputs;
    Blob<float>* output_layer = net_->output_blobs()[0];
    int j;
    for (j = 0; j < output_layer->num(); ++j) {
       const float* begin = output_layer->cpu_data() + j * output_layer->channels();
       const float* end = begin + output_layer->channels();
       /* Copy the output layer to a std::vector */
       outputs.push_back(std::vector<float>(begin, end));
    }
    return outputs;
}

void Classifier::WarmUp( )
{
  Mat img(input_geometry_.height * 4,input_geometry_.width * 8,CV_8UC3,Scalar(0,0,0));
  std::vector<std::vector<float> > outputs = PredictCuda(img);
}

/* By using Go as the HTTP server, we have potentially more CPU threads than
 * available GPUs and more threads can be added on the fly by the Go
 * runtime. Therefore we cannot pin the CPU threads to specific GPUs.  Instead,
 * when a CPU thread is ready for inference it will try to retrieve an
 * execution context from a queue of available GPU contexts and then do a
 * cudaSetDevice() to prepare for execution. Multiple contexts can be allocated
 * per GPU. */
class ExecContext
{
public:
    friend ScopedContext<ExecContext>;

    static bool IsCompatible(int device)
    {
        cudaError_t st = cudaSetDevice(device);
        if (st != cudaSuccess)
            return false;

        cuda::DeviceInfo info;
        if (!info.isCompatible())
            return false;

        return true;
    }

    ExecContext(const char* model_file,
                const char* trained_file,
                 const string& mean_file,
                 const string& label_file,
                 int device)
        : device_(device)
    {
        cudaError_t st = cudaSetDevice(device_);
        if (st != cudaSuccess)
            throw std::invalid_argument("could not set CUDA device");
        caffe_context_.reset(new Caffe);
        Caffe::Set(caffe_context_.get());
        classifier_.reset(new Classifier(model_file, trained_file,
                                         mean_file, label_file));
        Caffe::Set(nullptr);
    }

    Classifier* CaffeClassifier()
    {
        return classifier_.get();
    }

private:
    void Activate()
    {
        cudaError_t st = cudaSetDevice(device_);
        if (st != cudaSuccess)
            throw std::invalid_argument("could not set CUDA device");
  //      allocator_->reset();
        Caffe::Set(caffe_context_.get());
    }

    void Deactivate()
    {
        Caffe::Set(nullptr);
    }

private:
    int device_;
    std::unique_ptr<Caffe> caffe_context_;
    std::unique_ptr<Classifier> classifier_;
};

struct classifier_ctx
{
    ContextPool<ExecContext> pool;
};

struct classifier_ctxlist
{
    classifier_ctx **ctxs;
    int size;
};

/* Currently, 1 execution contexts are created per GPU. In other words, 2
 * inference tasks can execute in parallel on the same GPU. This helps improve
 * GPU utilization since some kernel operations of inference will not fully use
 * the GPU. */
constexpr static int kContextsPerDevice = 1;

classifier_ctxlist* classifier_initialize(char* model_file, char* trained_file,
                                      char* mean_file, char* label_file)
{
    try
    {
        ::google::InitGoogleLogging("inference_server");
        ::google::SetLogDestination(google::INFO,"");
        ::google::SetLogDestination(google::WARNING,"");
        ::google::SetLogDestination(google::ERROR,"");
        int device_count;
        cudaError_t st = cudaGetDeviceCount(&device_count);
        if (st != cudaSuccess)
            throw std::invalid_argument("could not list CUDA devices");
  
        std::vector<std::string> model_files;
        std::vector<std::string> trained_files;

        std::ifstream infile;
        infile.open(model_file);
        if(!infile.is_open())
            throw std::invalid_argument("could not open prototxt file");
        std::string s;
        while(getline(infile,s)){
            model_files.push_back(s);
        }
        infile.close();
        
        infile.open(trained_file);
        if(!infile.is_open())
            throw std::invalid_argument("could not open caffemodel file");
        while(getline(infile,s)){
            trained_files.push_back(s);
        }
        infile.close();
        
        if (model_files.size() != trained_files.size()){
           throw std::invalid_argument("number of prototxts is not equal to caffemodels");
        } 
        
        classifier_ctxlist* ctxlist = new classifier_ctxlist;
        ctxlist->size = model_files.size();
        ctxlist->ctxs = new classifier_ctx *[ctxlist->size];
 
        for (unsigned int j = 0; j < model_files.size(); ++j)
        {   
            ContextPool<ExecContext> pool;
            for (int dev = 0; dev < device_count; ++dev)
            {
                if (!ExecContext::IsCompatible(dev))
                {
                    LOG(ERROR) << "Skipping device: " << dev;
                    continue;
                }

                for (int i = 0; i < kContextsPerDevice; ++i)
                {
  //                  printf("model %d : %s\n",j,model_files[j].c_str());
                    std::unique_ptr<ExecContext> context(new ExecContext(model_files[j].c_str(), trained_files[j].c_str(),
                                                                       mean_file, label_file, dev));
                    pool.Push(std::move(context));
                }
            }

            if (pool.Size() == 0)
                throw std::invalid_argument("no suitable CUDA device");

            classifier_ctx* ctx = new classifier_ctx{std::move(pool)};
            ctxlist->ctxs[j] = ctx;
            /* Successful CUDA calls can set errno. */
            errno = 0;
        }
        return ctxlist;
    }
    catch (const std::invalid_argument& ex)
    {
        LOG(ERROR) << "exception: " << ex.what();
        errno = EINVAL;
        return nullptr;
    }
}

const char* classifier_classify(classifier_ctxlist* ctxlist,char* buffer)
{
    try
    {
        Mat img = imread(buffer);
        if (img.empty())
            throw std::invalid_argument("could not decode image");
        
        if (!img.isContinuous())
            img = img.clone();
        int h = img.rows;
        int w = img.cols;
        int size_in = min(w,h);
        int min = 0;
        int min_idx,diff;
        int size[3] = {1080,720,540};
        for(int s = 0; s < 3; ++s){
           diff = abs(size_in - size[s]); 
           if(s == 0 || (s > 0 && diff < min)){
              min  = diff;
              min_idx = s;
           }
        }
        classifier_ctx* ctx = ctxlist->ctxs[min_idx];
        std::vector<std::vector<Prediction> > all_predictions;
        /* In this scope an execution context is acquired for inference and it
        * will be automatically released back to the context pool when
        * exiting this scope. */
        ScopedContext<ExecContext> context(ctx->pool);
        auto classifier = context->CaffeClassifier();
        all_predictions = classifier->Classify(img);
        
        std::ostringstream os;
        double scores = 0;
        os << "[";
        for (size_t i = 0; i < all_predictions.size(); ++i) {
            std::vector<Prediction>& predictions = all_predictions[i];
            double score = 0;
//            os << "{\"tile" << i << "\":\"";
            for (size_t j = 0; j < predictions.size(); ++j){
                 Prediction p = predictions[j];
//                 os << p.first << "-" << p.second << " ";
                 if( j == 0 )
                   score += 0 * p.second;   
                 if( j == 1 )
                   score += 0.5 * p.second;   
                 if( j == 2 )
                   score += 1.0 * p.second;   
            }
  //          os << "\"},\n";
            if(score > 1)
               score = 1;
            scores += score;
        }
        scores = scores / all_predictions.size();
        os << "{\"score\":" << std::fixed << std::setprecision(6) << scores << "}";
        os << "]";

        errno = 0;
        std::string str = os.str();
        return strdup(str.c_str());
    }
    catch (const std::invalid_argument&)
    {
        errno = EINVAL;
        return nullptr;
    }
}

void classifier_destroy(classifier_ctxlist* ctxlist)
{   
    for(int i = 0; i < ctxlist->size; ++i)
        delete ctxlist->ctxs[i];
    ctxlist->size = 0;
    delete ctxlist;
}
