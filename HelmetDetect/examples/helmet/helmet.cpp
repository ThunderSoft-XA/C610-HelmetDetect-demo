#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <string>
#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include "opencv2/opencv.hpp"
#include "RunTflite/tflite_inference.h"

#include "gst_pipe/gstpipe.hpp"
#include "gst_pipe/gstpipefactory.hpp"

#include "pb_conf/gstreamer.pb.h"
#include "pb_conf/aiconf.pb.h"

#include "examples/param_parse.hpp"

using namespace std;
using namespace cv;

using namespace gstpipe;
using namespace ai2nference;

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;

#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

std::vector<gstpipe::GstPipe*> gst_pipe_vec;
std::vector <InferTFLITE *> ai_inference_vec;
cv::Mat* show_mat;

void preview(cv::Mat& imgframe)
{
    cv::Mat showframe;
    cv::cvtColor(imgframe,showframe,CV_BGR2RGBA);
    cv::imshow("sink", showframe);
    cv::waitKey(1);
    return ;
}

//导向滤波器
Mat guidedFilter(Mat &srcMat, Mat &guidedMat, int radius, double eps)
{
    //------------【0】转换源图像信息，将输入扩展为64位浮点型，以便以后做乘法------------
    srcMat.convertTo(srcMat, CV_64FC1);
    guidedMat.convertTo(guidedMat, CV_64FC1);
    //--------------【1】各种均值计算----------------------------------
    Mat mean_p, mean_I, mean_Ip, mean_II;
    boxFilter(srcMat, mean_p, CV_64FC1, Size(radius, radius));//生成待滤波图像均值mean_p 
    boxFilter(guidedMat, mean_I, CV_64FC1, Size(radius, radius));//生成引导图像均值mean_I   
    boxFilter(srcMat.mul(guidedMat), mean_Ip, CV_64FC1, Size(radius, radius));//生成互相关均值mean_Ip
    boxFilter(guidedMat.mul(guidedMat), mean_II, CV_64FC1, Size(radius, radius));//生成引导图像自相关均值mean_II
    //--------------【2】计算相关系数，计算Ip的协方差cov和I的方差var------------------
    Mat cov_Ip = mean_Ip - mean_I.mul(mean_p);
    Mat var_I = mean_II - mean_I.mul(mean_I);
    //---------------【3】计算参数系数a、b-------------------
    Mat a = cov_Ip / (var_I + eps);
    Mat b = mean_p - a.mul(mean_I);
    //--------------【4】计算系数a、b的均值-----------------
    Mat mean_a, mean_b;
    boxFilter(a, mean_a, CV_64FC1, Size(radius, radius));
    boxFilter(b, mean_b, CV_64FC1, Size(radius, radius));
    //---------------【5】生成输出矩阵------------------
    Mat dstImage = mean_a.mul(srcMat) + mean_b;
    return dstImage;
}

void bodyInference()
{
    std::cout << "enter body inference thread........" << std::endl; 
    for(;;) {
        for(auto ai_inference : ai_inference_vec) {
            if(ai_inference == nullptr) {
                continue;
            }

            cv::Mat helmet_mat;
            cv::Mat resultMat;  
            if ( false == gst_pipe_vec[0]->getFrameData(gst_pipe_vec[0],helmet_mat)) {
                continue;
            }
            vector<Mat> vSrcImage, vResultImage;
            split(helmet_mat, vSrcImage);
            for (int i = 0; i < 3; i++)
            {
                Mat tempImage;
                vSrcImage[i].convertTo(tempImage, CV_64FC1, 1.0 / 255.0);//将分通道转换成浮点型数据
                Mat cloneImage = tempImage.clone(); //将tempImage复制一份到cloneImage
                Mat resultImage = guidedFilter(tempImage, cloneImage, 5, 0.01);//对分通道分别进行导向滤波，半径为1、3、5...等奇数
                vResultImage.push_back(resultImage);//将分通道导向滤波后的结果存放到vResultImage中
            }
            merge(vResultImage, resultMat);
            // cv::cvtColor(helmet_mat_source, helmet_mat,cv::COLOR_BGR2RGB);    helmet tflite model need BGR style
            cv::Mat inputBlob;
        //    blobFromImagesFromOpencv(helmet_mat,inputBlob, 1 , cv::Size(260, 260), cv::Scalar(0, 0, 0), false,false,CV_8U);
            cv::Mat fact_mat;
            // cvtColor(helmet_mat, inputBlob, COLOR_BGR2RGB);
            cv::resize(resultMat,fact_mat,cv::Size(300,300),cv::INTER_LINEAR);
            cv::normalize(fact_mat,inputBlob,0,255,cv::NORM_MINMAX);
            // cv::convertScaleAbs(inputBlob,fact_mat,1.5,0);
            // helmet tflite model need BGR style
            vector<uchar> helmet_mat_vec = convertMat2Vector<uchar>(inputBlob);
            ai_inference->loadTfliteData<uchar>(helmet_mat_vec);
            std::vector<std::vector<float>> inference_result;
            ai_inference->doInference<float>(&inference_result);
            for(int idx = 0; idx < int(inference_result[3][0]); idx++) {
                if(inference_result[1][idx] == 0.0) {
                    // helmet
                    cv::Scalar color;
                    if( inference_result[2][idx] > 0.5) {
                        color = cv::Scalar(0,255,0);
                        cv::putText(helmet_mat,SSTR("helmet"),
                                                cv::Point2f(inference_result[0][idx * 4 + 1] * 640 + 10, inference_result[0][idx * 4] * 480  + 20),
                                                cv::FONT_HERSHEY_SIMPLEX, 0.8, color);
                    } else {
                        // without helmet
                        color = cv::Scalar(0,0,255);
                        cv::putText(helmet_mat,SSTR("without_helmet"),
                                                cv::Point2f(inference_result[0][idx * 4 + 1] * 640 + 10, inference_result[0][idx * 4] * 480  + 20),
                                                cv::FONT_HERSHEY_SIMPLEX, 0.8, color);
                    }
                    rectangle(helmet_mat,cv::Point2f(inference_result[0][idx * 4 + 1]* 640,inference_result[0][idx * 4]* 480),
                                        cv::Point2f(inference_result[0][idx * 4 + 3]* 640,inference_result[0][idx * 4 + 2]* 480), color, 1);
                }
            }
            show_mat = new cv::Mat(helmet_mat);
            // cv::imshow("sink", helmet_mat);
            // cv::waitKey(1);
        }
    }
}

void showMat() {
    cv::Mat* last_mat;
    for( ; ;) {
        if(nullptr == show_mat || last_mat == show_mat) {
            continue;
        }
        preview(*show_mat);
        last_mat = show_mat;
    }
}

int main(int argc, char ** argv)
{
    GMainLoop *main_loop = g_main_loop_new(NULL,false);

    // Verify that the version of the library that we linked against is
    // compatible with the version of the headers we compiled against.
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    parse::Parse parma_set = parse::parseArgs(argc,argv);

    if(parma_set.be_fill == false) {
        std::cout << "don`t parse required parma" << std::endl;
        parse::printHelp();
        return -1;
    }

    gstcfg::DataSourceSet data_source_set;
    aicfg::AISet ai_config_set;
    {
        // Read the existing config file.
        int fd = open(parma_set.config_file.c_str(), O_RDONLY);
        FileInputStream* input = new FileInputStream(fd);
        if (!google::protobuf::TextFormat::Parse(input, &data_source_set)) {
            cerr << "Failed to parse gstreamer data source." << endl;
            delete input;
            close(fd);
            return -1;
        }

        fd = open(parma_set.model_file.c_str(), O_RDONLY);
        input = new FileInputStream(fd);
        if (!google::protobuf::TextFormat::Parse(input, &ai_config_set)) {
            cerr << "Failed to parse gstreamer data source." << endl;
            delete input;
            close(fd);
            return -1;
        }
    }

    for(int i = 0; i < ai_config_set.config_size(); i++) {
        std::cout << "AI runtime config info " << i+1 << ",as follow:" <<std::endl;
        const aicfg::AIConfig& ai_config = ai_config_set.config(i);
        ai2nference::InferTFLITE* ai_inference = new InferTFLITE(ai_config.model_path(),(ai2nference::DataFormat)ai_config.data_format());
        ai_inference_vec.push_back(ai_inference);
        std::cout << "loading tflite model ......" <<std::endl;
        ai_inference->initRuntime((ai2nference::RunTime)ai_config.runtime());
    }

    std::cout << "tflite init runtime env finished ......" << std::endl;

    int stream_count = data_source_set.data_source_size();
    gstpipe::GstPipeFactory* pipe_factory = gstpipe::GstPipeFactory::getInstance();
    for (int i = 0; i < stream_count; i++) {
        std::cout << "gstreamer pipe config info " << i+1 << ",as follow:" <<std::endl;
        const gstcfg::DataSource& data_source = data_source_set.data_source(i);
        gstpipe::GstType gsttype = (gstpipe::GstType)data_source.gst_type();
        gstpipe::GstPipe* gst_pipe;
        gst_pipe = pipe_factory->createPipeLine(gsttype);

        gst_pipe->setPipeName(data_source.gst_name());
        gst_pipe->setSinkName(data_source.sink_name());
        gst_pipe->setGstType((GstType)data_source.gst_type());
        gst_pipe->setWidth(data_source.data_info().width());
        gst_pipe->setHeight(data_source.data_info().height());
        gst_pipe->setDecodeType(data_source.data_info().decode());
        gst_pipe->setFormat(data_source.data_info().format());
        gst_pipe->setFramerate(data_source.data_info().framerate());
        gst_pipe->setPath(data_source.gst_path());
        gst_pipe->setNeedCalib(data_source.neeed_calib());
        gst_pipe->setHwDec(data_source.enable_ai());

        gst_pipe_vec.emplace_back(gst_pipe);
    }

    for(auto gst_pipe : gst_pipe_vec) {
        gst_pipe->Init(argc,argv);
        std::thread gst_thread([=]{
            gst_pipe->runGst();
        });
        gst_thread.join();
    }
    std::cout << "all gstreamer pipe init and run finished ......" << std::endl;

    std::thread snpeInferenceThread(bodyInference);
    snpeInferenceThread.detach();

    std::thread showThread(showMat);
    showThread.join();

    // Optional:  Delete all global objects allocated by libprotobuf.
    google::protobuf::ShutdownProtobufLibrary();

    g_main_loop_run(main_loop);
    g_main_loop_unref(main_loop);

    return 0;
}
