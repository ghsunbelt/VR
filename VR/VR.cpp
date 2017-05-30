// VR_01.cpp : �������̨Ӧ�ó������ڵ㡣
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
* \��������
*   OtsuThreshold() 
* 
* \�������:
*   pGrayMat:      ��ֵͼ������
*   width:         ͼ�γߴ���
*   height:        ͼ�γߴ�߶�
*   nTlreshold:    �����㷨����õ��Ķ�ֵ���ָ���ֵ 
*   ostu_type:     һάostu���Ƕ�άostu
* \����ֵ: 
*   ��
* \����˵��:ʵ�ֻҶ�ͼ�Ķ�ֵ���ָ�������䷽���Otsu�㷨���׳ƴ���㷨��
* \����˵��:ʵ�ֻҶ�ͼ�Ķ�ֵ���ָ�������䷽�����άOtsu�㷨�� 
* \��ע���ڹ�����άֱ��ͼ��ʱ�򣬲��ûҶȵ��3*3�����ֵ 
* 
****************************************************************************/  
void OtsuThreshold(CvArr *pGrayMat, int width, int height, BYTE &nThreshold, THRESHOLD_TYPE ostu_type = OneDimentionOtsu )  
{  
	IplImage *pGrayImage = (IplImage *)pGrayMat;
	switch(ostu_type)
	{
	case TwoDimentionOtsu:
		{
			double dHistogram[256][256];        //������ά�Ҷ�ֱ��ͼ  
			double dTrMatrix = 0.0;             //��ɢ����ļ�  
			int N = height*width;               //��������  
			for(int i=0; i<256; i++)  
			{  
				for(int j=0; j<256; j++)  
					dHistogram[i][j] = 0.0;      //��ʼ������  
			}  
			for(int i=0; i<height; i++)  
			{  
				for(int j=0; j<width; j++)  
				{  
					BYTE nData1 = CV_IMAGE_ELEM(pGrayImage,BYTE, i, j);  //��ǰ�ĻҶ�ֵ  
					BYTE nData2 = 0;  
					int nData3 = 0;         //ע��9��ֵ��ӿ��ܳ���һ���ֽ�  
					for(int m=i-1; m<=i+1; m++)  
					{  
						for(int n=j-1; n<=j+1; n++)  
						{  
							if((m>=0)&&(m<height)&&(n>=0)&&(n<width))  
								nData3 += CV_IMAGE_ELEM(pGrayImage,BYTE, m, n); //��ǰ�ĻҶ�ֵ  
						}  
					}  
					nData2 = (BYTE)(nData3/9);    //����Խ�������ֵ���в���,�����ֵ  
					dHistogram[nData1][nData2]++; 
					//cout << i << " " << j<< endl;
				}  
			}  
			for(int i=0; i<256; i++)  
				for(int j=0; j<256; j++)  
					dHistogram[i][j] /= N;  //�õ���һ���ĸ��ʷֲ�  

			double Pai = 0.0;      //Ŀ������ֵʸ��i����  
			double Paj = 0.0;      //Ŀ������ֵʸ��j����  
			double Pbi = 0.0;      //��������ֵʸ��i����  
			double Pbj = 0.0;      //��������ֵʸ��j����  
			double Pti = 0.0;      //ȫ�־�ֵʸ��i����  
			double Ptj = 0.0;      //ȫ�־�ֵʸ��j����  
			double W0 = 0.0;       //Ŀ���������ϸ����ܶ�  
			double W1 = 0.0;       //�����������ϸ����ܶ�  
			double dData1 = 0.0;  
			double dData2 = 0.0;  
			double dData3 = 0.0;  
			double dData4 = 0.0;   //�м����  
			int nThreshold_s = 0;  
			int nThreshold_t = 0;  
			double temp = 0.0;     //Ѱ�����ֵ  
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
					/*          W1=dData3=dData4=0.0;   //����ѭ�������ݽ��г�ʼ�� 
					for(int s=i+1; s<256; s++) 
					{ 
					for(int t=j+1; t<256; t++) 
					{ 
					W1 += dHistogram[s][t]; 
					dData3 += s*dHistogram[s][t];  //����2 
					dData4 += t*dHistogram[s][t];  //Ҳ��������ѭ�����м��� 
					} 
					}*/  

					Pai = dData1/W0;  
					Paj = dData2/W0;  
					Pbi = dData3/W1;  
					Pbj = dData4/W1;   // �õ�������ֵ��������4��������ʾ  
					dTrMatrix = ((W0*Pti-dData1)*(W0*Pti-dData1)+(W0*Ptj-dData1)*(W0*Ptj-dData2))/(W0*W1);  
					if(dTrMatrix > temp)  
					{  
						temp = dTrMatrix;  
						nThreshold_s = i;  
						nThreshold_t = j;  
					}  
				}  
			}  
			nThreshold = nThreshold_t;   //���ؽ���еĻҶ�ֵ  
			//nThreshold = 100;  
		}
		break;
	case OneDimentionOtsu:
		{
			double nHistogram[256];         //�Ҷ�ֱ��ͼ  
			double dVariance[256];          //��䷽��  
			int N = height*width;           //��������  
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
					nHistogram[nData]++;     //����ֱ��ͼ  
				}  
			}  
			double Pa=0.0;      //�������ָ���  
			double Pb=0.0;      //Ŀ����ָ���  
			double Wa=0.0;      //����ƽ���Ҷ�ֵ  
			double Wb=0.0;      //Ŀ��ƽ���Ҷ�ֵ  
			double W0=0.0;      //ȫ��ƽ���Ҷ�ֵ  
			double dData1=0.0,  dData2=0.0;  
			for(int i=0; i<256; i++)     //����ȫ��ƽ���Ҷ�  
			{  
				nHistogram[i] /= N;  
				W0 += i*nHistogram[i];  
			}  
			for(int i=0; i<256; i++)     //��ÿ���Ҷ�ֵ������䷽��  
			{  
				Pa += nHistogram[i];  
				Pb = 1-Pa;  
				dData1 += i*nHistogram[i];  
				dData2 = W0-dData1;  
				Wa = dData1/Pa;  
				Wb = dData2/Pb;  
				dVariance[i] = (Pa*Pb* pow((Wb-Wa), 2));  
			}  
			//����ÿ�������ȡ�����󷽲�����Ӧ�ĻҶ�ֵ  
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
			// nMaxIter��������������nDiffRec��ʹ�ø�����ֵȷ���������밵��ƽ���ҶȲ���ֵ  
			//int DetectThreshold(IplImage*pGrayMat, int nMaxIter, int& iDiffRec) //��ֵ�ָ������   
			//ͼ����Ϣ 
			int nMaxIter = 10;
			int iDiffRec =  10;  
			int F[256]={ 0 }; //ֱ��ͼ����  
			unsigned long long iTotalGray=0;//�Ҷ�ֵ��  
			unsigned long long iTotalPixel =0;//��������  
			BYTE bt;//ĳ�������ֵ  

			uchar iNewThrehold;//��ֵ���·�ֵ  
			uchar iMaxGrayValue=0,iMinGrayValue=255;//ԭͼ���е����Ҷ�ֵ����С�Ҷ�ֵ  
			uchar iMeanGrayValue1,iMeanGrayValue2;  

			//��ȡ(i,j)��ֵ������ֱ��ͼ����F  
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
			iNewThrehold = (iMinGrayValue+iMaxGrayValue)/2;//��ʼ��ֵ  
			iDiffRec = iMaxGrayValue - iMinGrayValue;  

			for(int a=0;(abs(nThreshold-iNewThrehold)>0.5)&&a<nMaxIter; a++)//������ֹ����  
			{  
				nThreshold = iNewThrehold;  
				//С�ڵ�ǰ��ֵ���ֵ�ƽ���Ҷ�ֵ  
				for(int i=iMinGrayValue;i<nThreshold;i++)  
				{  
					iTotalGray += F[i]*i;//F[]�洢ͼ����Ϣ  
					iTotalPixel += F[i];  
				}  
				iMeanGrayValue1 = (uchar)(iTotalGray/iTotalPixel);  
				//���ڵ�ǰ��ֵ���ֵ�ƽ���Ҷ�ֵ  
				iTotalPixel =0;  
				iTotalGray =0;  
				for(int j=nThreshold+1;j<iMaxGrayValue;j++)  
				{  
					iTotalGray += F[j]*j;//F[]�洢ͼ����Ϣ
					iTotalPixel += F[j];   
				}  
				iMeanGrayValue2 = (uchar)(iTotalGray/iTotalPixel);  
				cout << (int)nThreshold <<" "<<(int)iNewThrehold << " "<< (int)iMeanGrayValue2 << " " << (int)iMeanGrayValue1<< endl;
				iNewThrehold = (iMeanGrayValue2+iMeanGrayValue1)/2; //�·�ֵ  
				iDiffRec = abs(iMeanGrayValue2 - iMeanGrayValue1);  
			}  	
		}  
		break;
	}
	cout<<"The Threshold of this Image in pGrayMatIteration is:"<<(int)nThreshold<<endl;   
}

//�����ȹ��յĲ�������
//1����ȡԴͼI��ƽ���Ҷȣ�����¼rows��cols��
//2������һ����С����ΪN*M�����飬���ÿ���ƽ��ֵ���õ��ӿ�����Ⱦ���D��
//3���þ���D��ÿ��Ԫ�ؼ�ȥԴͼ��ƽ���Ҷȣ��õ��ӿ�����Ȳ�ֵ����E��
//4����˫������ֵ����������E��ֵ����Դͼһ����С�����ȷֲ�����R��
//5���õ��������ͼ��result=I-R��
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
	//��������
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
	//�����滻��ͼƬ
	IplImage* image;
	if(strImage == "")strImage = "./Images/123.jpg";
	image = cvLoadImage(strImage);
	assert(image != NULL);
	cvShowImage("image", image);
	//����Ƶ
	CvCapture* capture;
	double fps = 0; //Ƶ��
	CvSize size; //��С
	if(strVideo == "")
		capture = cvCreateCameraCapture(0);   //0Ϊ�ⲿ����ͷ��ID��1Ϊ�ʼǱ���������ͷ��ID
	else
		capture = cvCreateFileCapture(strVideo);
	assert(capture != NULL);
	fps = cvGetCaptureProperty(capture,CV_CAP_PROP_FPS);
	size = cvSize((int)cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_WIDTH), (int)cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_HEIGHT));
	strcat(strVideo,"_v0.1.avi");
	//д��Ƶ
	CvVideoWriter *writer = cvCreateVideoWriter(strVideo,CV_FOURCC('M','P','4','2'),fps,size);
	assert(writer != NULL);
	//��ȡ��Ƶ
	IplImage* src0 = NULL;
	IplImage* src = NULL;
	IplImage *grayImage = cvCreateImage(size, 8, 1);//�����Ҷ�ͼ
	CvMemStorage * storage = cvCreateMemStorage(0);
	for (;;src0 =NULL)
	{
		if((src0 = cvQueryFrame(capture)) == NULL)break;
		src = src0;
		cvSmooth(src0,src,CV_GAUSSIAN,3,3,0,0); 
		//��ɫת�Ҷ�
		cvCvtColor(src, grayImage, CV_RGB2GRAY);
		//����Ӱ����
		Mat *grayMat= new Mat(grayImage, 0 );
		////imshow("grayMat11", *grayMat);
		unevenLightCompensate(*grayMat,1<<12);
		imshow("grayMat", *grayMat);
		//��ֵ��
		//BYTE thre;//��ֵ����ֵ
		//OtsuThreshold(graypGrayMat, graypGrayMat->width,graypGrayMat->height, thre, IterativeMethod);//������ֵ
		//cvThreshold(graypGrayMat, graypGrayMat, thre, 255, CV_THRESH_BINARY);//��ֵ�� 
		cvThreshold(grayImage,grayImage,0,255,cv::THRESH_OTSU);
		//��ʴ
		cvErode(grayImage,grayImage); 
		// ͼ������
		CvSeq * contour = 0;
		int num = cvFindContours(grayImage, storage, &contour, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		CvPoint2D32f pt1[4] = {0,image->height-1,0,0,image->width-1,0,image->width-1,image->height-1};
		//printf("origin  pos:(%3f,%3f),(%3f,%3f),(%3f,%3f)\n",pt1[0].x, pt1[0].y, pt1[1].x, pt1[1].y, pt1[2].x, pt1[2].y);
		CvPoint2D32f pt2[4];
		int area = 0;
		// ������������
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

