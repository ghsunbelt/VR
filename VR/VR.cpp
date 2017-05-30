// VR_01.cpp : 定义控制台应用程序的入口点。
//
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <tchar.h>

using namespace std;
using namespace cv;

#define BYTE unsigned char
inline CvPoint CvPointf2d(CvPoint2D32f &fPoint){CvPoint dPoint; dPoint.x = (int)fPoint.x; dPoint.y = (int)fPoint.y; return dPoint;}

enum THRESHOLD_TYPE{OneDimentionOtsu, TwoDimentionOtsu,  IterativeMethod};

/***************************************************************************** 
* 
* \函数名称
*   OtsuThreshold() 
* 
* \输入参数:
*   pGrayMat:      二值图像数据
*   width:         图形尺寸宽度
*   height:        图形尺寸高度
*   nTlreshold:    经过算法处理得到的二值化分割阈值 
*   ostu_type:     一维ostu还是二维ostu
* \返回值: 
*   无
* \函数说明:实现灰度图的二值化分割——最大类间方差法（Otsu算法，俗称大津算法）
* \函数说明:实现灰度图的二值化分割——最大类间方差法（二维Otsu算法） 
* \备注：在构建二维直方图的时候，采用灰度点的3*3邻域均值 
* 
****************************************************************************/  
void OtsuThreshold(CvArr *pGrayMat, int width, int height, BYTE &nThreshold, THRESHOLD_TYPE ostu_type = OneDimentionOtsu )  
{  
	IplImage *pGrayImage = (IplImage *)pGrayMat;
	switch(ostu_type)
	{
	case TwoDimentionOtsu:
		{
			double dHistogram[256][256];        //建立二维灰度直方图  
			double dTrMatrix = 0.0;             //离散矩阵的迹  
			int N = height*width;               //总像素数  
			for(int i=0; i<256; i++)  
			{  
				for(int j=0; j<256; j++)  
					dHistogram[i][j] = 0.0;      //初始化变量  
			}  
			for(int i=0; i<height; i++)  
			{  
				for(int j=0; j<width; j++)  
				{  
					BYTE nData1 = CV_IMAGE_ELEM(pGrayImage,BYTE, i, j);  //当前的灰度值  
					BYTE nData2 = 0;  
					int nData3 = 0;         //注意9个值相加可能超过一个字节  
					for(int m=i-1; m<=i+1; m++)  
					{  
						for(int n=j-1; n<=j+1; n++)  
						{  
							if((m>=0)&&(m<height)&&(n>=0)&&(n<width))  
								nData3 += CV_IMAGE_ELEM(pGrayImage,BYTE, m, n); //当前的灰度值  
						}  
					}  
					nData2 = (BYTE)(nData3/9);    //对于越界的索引值进行补零,邻域均值  
					dHistogram[nData1][nData2]++; 
					//cout << i << " " << j<< endl;
				}  
			}  
			for(int i=0; i<256; i++)  
				for(int j=0; j<256; j++)  
					dHistogram[i][j] /= N;  //得到归一化的概率分布  

			double Pai = 0.0;      //目标区均值矢量i分量  
			double Paj = 0.0;      //目标区均值矢量j分量  
			double Pbi = 0.0;      //背景区均值矢量i分量  
			double Pbj = 0.0;      //背景区均值矢量j分量  
			double Pti = 0.0;      //全局均值矢量i分量  
			double Ptj = 0.0;      //全局均值矢量j分量  
			double W0 = 0.0;       //目标区的联合概率密度  
			double W1 = 0.0;       //背景区的联合概率密度  
			double dData1 = 0.0;  
			double dData2 = 0.0;  
			double dData3 = 0.0;  
			double dData4 = 0.0;   //中间变量  
			int nThreshold_s = 0;  
			int nThreshold_t = 0;  
			double temp = 0.0;     //寻求最大值  
			for(int i=0; i<256; i++)  
			{  
				for(int j=0; j<256; j++)  
				{  
					Pti += i*dHistogram[i][j];  
					Ptj += j*dHistogram[i][j];  
				}  
			}  
			for(int i=0; i<256; i++)  
			{  
				for(int j=0; j<256; j++)  
				{  
					W0 += dHistogram[i][j];  
					dData1 += i*dHistogram[i][j];  
					dData2 += j*dHistogram[i][j];  

					W1 = 1-W0;  
					dData3 = Pti-dData1;  
					dData4 = Ptj-dData2;  
					/*          W1=dData3=dData4=0.0;   //对内循环的数据进行初始化 
					for(int s=i+1; s<256; s++) 
					{ 
					for(int t=j+1; t<256; t++) 
					{ 
					W1 += dHistogram[s][t]; 
					dData3 += s*dHistogram[s][t];  //方法2 
					dData4 += t*dHistogram[s][t];  //也可以增加循环进行计算 
					} 
					}*/  

					Pai = dData1/W0;  
					Paj = dData2/W0;  
					Pbi = dData3/W1;  
					Pbj = dData4/W1;   // 得到两个均值向量，用4个分量表示  
					dTrMatrix = ((W0*Pti-dData1)*(W0*Pti-dData1)+(W0*Ptj-dData1)*(W0*Ptj-dData2))/(W0*W1);  
					if(dTrMatrix > temp)  
					{  
						temp = dTrMatrix;  
						nThreshold_s = i;  
						nThreshold_t = j;  
					}  
				}  
			}  
			nThreshold = nThreshold_t;   //返回结果中的灰度值  
			//nThreshold = 100;  
		}
		break;
	case OneDimentionOtsu:
		{
			double nHistogram[256];         //灰度直方图  
			double dVariance[256];          //类间方差  
			int N = height*width;           //总像素数  
			for(int i=0; i<256; i++)  
			{  
				nHistogram[i] = 0.0;  
				dVariance[i] = 0.0;  
			}  
			for(int i=0; i<height; i++)  
			{  
				for(int j=0; j<width; j++)  
				{  
					BYTE nData = CV_IMAGE_ELEM(pGrayImage,BYTE, i, j);
					nHistogram[nData]++;     //建立直方图  
				}  
			}  
			double Pa=0.0;      //背景出现概率  
			double Pb=0.0;      //目标出现概率  
			double Wa=0.0;      //背景平均灰度值  
			double Wb=0.0;      //目标平均灰度值  
			double W0=0.0;      //全局平均灰度值  
			double dData1=0.0,  dData2=0.0;  
			for(int i=0; i<256; i++)     //计算全局平均灰度  
			{  
				nHistogram[i] /= N;  
				W0 += i*nHistogram[i];  
			}  
			for(int i=0; i<256; i++)     //对每个灰度值计算类间方差  
			{  
				Pa += nHistogram[i];  
				Pb = 1-Pa;  
				dData1 += i*nHistogram[i];  
				dData2 = W0-dData1;  
				Wa = dData1/Pa;  
				Wb = dData2/Pb;  
				dVariance[i] = (Pa*Pb* pow((Wb-Wa), 2));  
			}  
			//遍历每个方差，求取类间最大方差所对应的灰度值  
			double temp=0.0;  
			for(int i=0; i<256; i++)  
			{  
				if(dVariance[i]>temp)  
				{  
					temp = dVariance[i];  
					nThreshold = i;  
				}  
			}  
		}
		break;
	case IterativeMethod:
		{
			// nMaxIter：最大迭代次数；nDiffRec：使用给定阀值确定的亮区与暗区平均灰度差异值  
			//int DetectThreshold(IplImage*pGrayMat, int nMaxIter, int& iDiffRec) //阀值分割：迭代法   
			//图像信息 
			int nMaxIter = 10;
			int iDiffRec =  10;  
			int F[256]={ 0 }; //直方图数组  
			unsigned long long iTotalGray=0;//灰度值和  
			unsigned long long iTotalPixel =0;//像素数和  
			BYTE bt;//某点的像素值  

			uchar iNewThrehold;//阀值、新阀值  
			uchar iMaxGrayValue=0,iMinGrayValue=255;//原图像中的最大灰度值和最小灰度值  
			uchar iMeanGrayValue1,iMeanGrayValue2;  

			//获取(i,j)的值，存于直方图数组F  
			for(int i=0;i<height;i++)  
			{  
				for(int j=0;j<width;j++)  
				{  
					bt = CV_IMAGE_ELEM(pGrayImage, BYTE, i, j);  
					if(bt<iMinGrayValue)  
						iMinGrayValue = bt;  
					if(bt>iMaxGrayValue)  
						iMaxGrayValue = bt;  
					F[bt]++;  
				}  
			}  

			nThreshold =0;//  
			iNewThrehold = (iMinGrayValue+iMaxGrayValue)/2;//初始阀值  
			iDiffRec = iMaxGrayValue - iMinGrayValue;  

			for(int a=0;(abs(nThreshold-iNewThrehold)>0.5)&&a<nMaxIter; a++)//迭代中止条件  
			{  
				nThreshold = iNewThrehold;  
				//小于当前阀值部分的平均灰度值  
				for(int i=iMinGrayValue;i<nThreshold;i++)  
				{  
					iTotalGray += F[i]*i;//F[]存储图像信息  
					iTotalPixel += F[i];  
				}  
				iMeanGrayValue1 = (uchar)(iTotalGray/iTotalPixel);  
				//大于当前阀值部分的平均灰度值  
				iTotalPixel =0;  
				iTotalGray =0;  
				for(int j=nThreshold+1;j<iMaxGrayValue;j++)  
				{  
					iTotalGray += F[j]*j;//F[]存储图像信息
					iTotalPixel += F[j];   
				}  
				iMeanGrayValue2 = (uchar)(iTotalGray/iTotalPixel);  
				cout << (int)nThreshold <<" "<<(int)iNewThrehold << " "<< (int)iMeanGrayValue2 << " " << (int)iMeanGrayValue1<< endl;
				iNewThrehold = (iMeanGrayValue2+iMeanGrayValue1)/2; //新阀值  
				iDiffRec = abs(iMeanGrayValue2 - iMeanGrayValue1);  
			}  	
		}  
		break;
	}
	cout<<"The Threshold of this Image in pGrayMatIteration is:"<<(int)nThreshold<<endl;   
}

//不均匀光照的补偿方法
//1、求取源图I的平均灰度，并记录rows和cols；
//2、按照一定大小，分为N*M个方块，求出每块的平均值，得到子块的亮度矩阵D；
//3、用矩阵D的每个元素减去源图的平均灰度，得到子块的亮度差值矩阵E；
//4、用双立方差值法，将矩阵E差值成与源图一样大小的亮度分布矩阵R；
//5、得到矫正后的图像result=I-R；
void unevenLightCompensate(Mat &image, int blockSize)
{
	double average = mean(image)[0];
	int rows_new = ceil(double(image.rows) / double(blockSize));
	int cols_new = ceil(double(image.cols) / double(blockSize));
	Mat blockImage;
	blockImage = Mat::zeros(rows_new, cols_new, CV_32FC1);
	for (int i = 0; i < rows_new; i++)
	{
		for (int j = 0; j < cols_new; j++)
		{
			int rowmin = i*blockSize;
			int rowmax = (i + 1)*blockSize;
			if (rowmax > image.rows) rowmax = image.rows;
			int colmin = j*blockSize;
			int colmax = (j + 1)*blockSize;
			if (colmax > image.cols) colmax = image.cols;
			Mat imageROI = image(Range(rowmin, rowmax), Range(colmin, colmax));
			double temaver = mean(imageROI)[0];
			blockImage.at<float>(i, j) = temaver;
		}
	}
	blockImage = blockImage - average;
	Mat blockImage2;
	resize(blockImage, blockImage2, image.size(), (0, 0), (0, 0), INTER_CUBIC);
	Mat image2;
	image.convertTo(image2, CV_32FC1);
	Mat dst = image2 - blockImage2;
	dst.convertTo(image, CV_8UC1);
}

int _tmain(int argc, _TCHAR* argv[])
{
	//参数处理
	_TCHAR *strVideo = "";
	_TCHAR *strImage = "";
	switch (argc)
	{
	case 1:
		break;
	case 2:
		strVideo = argv[1];
		break;
	case 3:
		strVideo = argv[1];
		strImage = argv[2];
		break;
	default:
		{
			printf("Use description:\n\tVR_01_1 [video] [image]\n");
			getchar();
			return 0;
		}
	}
	//用来替换的图片
	IplImage* image;
	if(strImage == "")strImage = "./Images/123.jpg";
	image = cvLoadImage(strImage);
	assert(image != NULL);
	cvShowImage("image", image);
	//读视频
	CvCapture* capture;
	double fps = 0; //频率
	CvSize size; //大小
	if(strVideo == "")
		capture = cvCreateCameraCapture(0);   //0为外部摄像头的ID，1为笔记本内置摄像头的ID
	else
		capture = cvCreateFileCapture(strVideo);
	assert(capture != NULL);
	fps = cvGetCaptureProperty(capture,CV_CAP_PROP_FPS);
	size = cvSize((int)cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_WIDTH), (int)cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_HEIGHT));
	strcat(strVideo,"_v0.1.avi");
	//写视频
	CvVideoWriter *writer = cvCreateVideoWriter(strVideo,CV_FOURCC('M','P','4','2'),fps,size);
	assert(writer != NULL);
	//读取视频
	IplImage* src0 = NULL;
	IplImage* src = NULL;
	IplImage *grayImage = cvCreateImage(size, 8, 1);//创建灰度图
	CvMemStorage * storage = cvCreateMemStorage(0);
	for (;;src0 =NULL)
	{
		if((src0 = cvQueryFrame(capture)) == NULL)break;
		src = src0;
		cvSmooth(src0,src,CV_GAUSSIAN,3,3,0,0); 
		//彩色转灰度
		cvCvtColor(src, grayImage, CV_RGB2GRAY);
		//光照影像处理
		Mat *grayMat= new Mat(grayImage, 0 );
		////imshow("grayMat11", *grayMat);
		unevenLightCompensate(*grayMat,1<<12);
		imshow("grayMat", *grayMat);
		//二值化
		//BYTE thre;//二值化阈值
		//OtsuThreshold(graypGrayMat, graypGrayMat->width,graypGrayMat->height, thre, IterativeMethod);//查找阈值
		//cvThreshold(graypGrayMat, graypGrayMat, thre, 255, CV_THRESH_BINARY);//二值化 
		cvThreshold(grayImage,grayImage,0,255,cv::THRESH_OTSU);
		//腐蚀
		cvErode(grayImage,grayImage); 
		// 图像轮廓
		CvSeq * contour = 0;
		int num = cvFindContours(grayImage, storage, &contour, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		CvPoint2D32f pt1[4] = {0,image->height-1,0,0,image->width-1,0,image->width-1,image->height-1};
		//printf("origin  pos:(%3f,%3f),(%3f,%3f),(%3f,%3f)\n",pt1[0].x, pt1[0].y, pt1[1].x, pt1[1].y, pt1[2].x, pt1[2].y);
		CvPoint2D32f pt2[4];
		int area = 0;
		// 查找最大的轮廓
		for (; contour != 0; contour = contour->h_next)
		{
			CvRect rect = ((CvContour *)contour)->rect;
			if (rect.width*rect.height>area)
			{
				area = rect.width*rect.height;
				//cvRectangle(src, cvPoint(rect.x, rect.y), cvPoint(rect.x + rect.width, rect.y + rect.height), CV_RGB(255, 0, 0), 1, CV_AA, 0);
			}
			else continue;

			CvBox2D box = cvMinAreaRect2(contour,0);
			if(box.size.height == 0 || box.size.width == 0 )continue;
			cvBoxPoints(box, pt2);
		}
		cvLine( grayImage, CvPointf2d(pt2[0]), CvPointf2d(pt2[1]),CV_RGB(0,0,255), 2, 8, 0 );
		cvLine( grayImage, CvPointf2d(pt2[1]), CvPointf2d(pt2[2]),CV_RGB(0,0,255), 2, 8, 0 );
		cvLine( grayImage, CvPointf2d(pt2[2]), CvPointf2d(pt2[3]),CV_RGB(0,0,255), 2, 8, 0 );
		cvLine( grayImage, CvPointf2d(pt2[3]), CvPointf2d(pt2[0]),CV_RGB(0,0,255), 2, 8, 0 );

		CvMat *transMat = cvCreateMat(2,3,CV_32F);
		cvGetAffineTransform(pt1, pt2, transMat);
		printf("transform mat:[%3f,%3f,%3f, %3f,%3f,%3f]\n", CV_MAT_ELEM(*transMat,float,0,0),CV_MAT_ELEM(*transMat,float,0,1),CV_MAT_ELEM(*transMat,float,0,2),\
			CV_MAT_ELEM(*transMat,float,1,0),CV_MAT_ELEM(*transMat,float,1,1),CV_MAT_ELEM(*transMat,float,1,2));

		/*cvMat *mat = cvCreateMat(2,3,CV_32F);
		mat->data<float>(0,0) = 0;
		mat->data<float>(0,1) = image->width-1,0,0,image->height-1,0,image->height-1,image->width-1
		*/

		IplImage *mastImage = cvCreateImage(cvGetSize(src),src->depth,src->nChannels);
		cvWarpAffine(image,mastImage,transMat);
		//cvShowImage("Mast",mastImage);
		//cvShowImage("Src",src);
		for(int i = 0;i<src->height;i++)
		{
			for(int j=0;j<src->width;j++)
			{
				CvPoint2D32f tmp = {i,j};
				//cout<<CV_IMAGE_ELEM(mastImage,Vec3b,i,j);
				if(CV_IMAGE_ELEM(mastImage,Vec3b,i,j)!= Vec3b(0,0,0))
				{
					CV_IMAGE_ELEM(src,Vec3b,i,j) = CV_IMAGE_ELEM(mastImage,Vec3b,i,j);//CV_IMAGE_ELEM(mastImage,Vec3b,i,j);
				}
			}
		}
		//cout<<"### ("<<src->height<<","<<src->width<<") ("<<graypGrayMat->height<<","<<graypGrayMat->width<<")"<<endl;
		cvSmooth(src,src,CV_MEDIAN,3,3);
		cvWriteFrame(writer,src);
		cvShowImage("Gray", grayImage);
		cvShowImage("Input", src);
		cvReleaseImage(&mastImage);
		
		char c = waitKey(50);
		if (c == 27) break;
	}
	cvReleaseVideoWriter(&writer);
	cvReleaseCapture(&capture);
	cvReleaseImage(&image);
	cvReleaseImage(&grayImage);
	cvReleaseMemStorage(&storage);
	return 0;
}

