//https://medium.com/analytics-vidhya/hand-detection-and-finger-counting-using-opencv-python-5b594704eb08
//https://answers.opencv.org/question/89927/how-to-use-knearest-in-java/
package com.first_java_app.opencvtutorial;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.widget.Toast;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.KNearest;
import org.opencv.ml.Ml;
import org.opencv.utils.Converters;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // Here, thisActivity is the current activity
        //----------read and write permission----------------
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 1);
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 1);
        }
        //----------END read and write permission----------------
        setContentView(R.layout.activity_main);

        if(OpenCVLoader.initDebug()){
            Toast.makeText(getApplicationContext(), "OpenCV loaded succesfully",Toast.LENGTH_LONG).show();
        }else{
            Toast.makeText(getApplicationContext(), "Couldn't load OpenCV",Toast.LENGTH_LONG).show();
        }
//        String imgInput = "/storage/emulated/0/Pictures/0.jpg";
//        Scalar lower = new Scalar(0, 0.02*255, 0);  // to pick hands color range
//        Scalar upper = new Scalar(30, 0.68*255, 255);
//        Mat samples = new Mat();
//        Mat skinRegionHSV = new Mat();
//        Mat blurred = new Mat();
//        Mat thresh = new Mat();
//        Imgproc.cvtColor(Imgcodecs.imread(imgInput), samples, Imgproc.COLOR_BGR2HSV);
//        Core.inRange(samples, lower, upper, skinRegionHSV);
//        Imgproc.blur(skinRegionHSV, blurred, new org.opencv.core.Size(2, 2));
//        Imgproc.threshold(blurred, thresh, 0, 255, Imgproc.THRESH_BINARY);

//        //----------save img to phone: input: thresh---------------
//        String filename = "thresh1.jpg";
//        Boolean bool = null;
//        File file = new File("/storage/emulated/0/Pictures", filename);
//        filename = file.toString();
//        bool = Imgcodecs.imwrite(filename, skinRegionHSV);
//        if (bool == true)
//            System.out.println("SUCCESS writing image to external storage");
//        else
//            System.out.println("Fail writing image to external storage");
//        //----------END save img to phone---------------


        Mat trainData = new Mat(),
                testData = new Mat();
        List<Integer> trainLabs = new ArrayList<Integer>(),
                testLabs = new ArrayList<Integer>();
        for(int r=0; r<2;r++){  // 5 signs  , r<2 for testing purpose, r<5 is the origin
            for(int c=0; c<40;c++){ // 40 samples per sign, sample 40 for test purpose
                //H,S,V area
//                Scalar lower = new Scalar(0, 0.28*255, 0);  // to pick hands color range
//                Scalar upper = new Scalar(25, 0.68*255, 255);
                Scalar lower = new Scalar(0, 0.02*255, 0);  // to pick hands color range
                Scalar upper = new Scalar(30, 0.68*255, 255);
                String imgInput = "/storage/emulated/0/Pictures/"+r+"/"+c+".jpg";

                Mat samples = new Mat();
                Mat skinRegionHSV = new Mat();
                Mat blurred = new Mat();
                Mat thresh = new Mat();
                //----------SkinMask processing------------
                Imgproc.cvtColor(Imgcodecs.imread(imgInput), samples, Imgproc.COLOR_BGR2HSV);
                Core.inRange(samples, lower, upper, skinRegionHSV);
                Imgproc.blur(skinRegionHSV, blurred, new org.opencv.core.Size(2, 2));
                Imgproc.threshold(blurred, thresh, 0, 255, Imgproc.THRESH_BINARY);
                //----------END SkinMask processing------------

                //----------prepare to train using knn----------
                // we need float data for knn:
                thresh.convertTo(thresh, CvType.CV_32F);
//                //-----split 50/50 to test-------
//                if (c % 2 == 0) {
//                    // for opencv ml, each feature has to be a single row:
//                    trainData.push_back(thresh.reshape(1,1));
//                    // add a label for that feature (the digit number):
//                    trainLabs.add(r);
//                } else {
//                    testData.push_back(thresh.reshape(1,1));
//                    testLabs.add(r);
//                }

                // for opencv ml, each feature has to be a single row:
                trainData.push_back(thresh.reshape(1,1));
                // add a label for that feature (the digit number):
                trainLabs.add(r);
                //----------END prepare to train using knn----------
            }
        }
        //------------train knn object----------
        // make a Mat of the train labels, and train knn:
        KNearest knn = KNearest.create();
        knn.train(trainData, Ml.ROW_SAMPLE, Converters.vector_int_to_Mat(trainLabs));
        //------------END train knn object----------

//        // now test predictions:
//        for (int i=0; i<testData.rows(); i++)
//        {
//            Mat one_feature = testData.row(i);
//            int testLabel = testLabs.get(i);
//
//            Mat res = new Mat();
//            float p = knn.findNearest(one_feature, 1, res);
//            System.out.println(testLabel + " " + p + " " + res.dump());
//            //System.out: 9 9.0 [9]
//        }

//        public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
//            Scalar lower = new Scalar(0, 0.28*255, 0);  // to pick hands color range
//            Scalar upper = new Scalar(25, 0.68*255, 255);
//            String imgInput = inputFrame;
//
//            Mat samples = new Mat();
//            Mat skinRegionHSV = new Mat();
//            Mat blurred = new Mat();
//            Mat thresh = new Mat();
//            //----------SkinMask processing------------
//            Imgproc.cvtColor(Imgcodecs.imread(imgInput), samples, Imgproc.COLOR_BGR2HSV);
//            Core.inRange(samples, lower, upper, skinRegionHSV);
//            Imgproc.blur(skinRegionHSV, blurred, new org.opencv.core.Size(2, 2));
//            Imgproc.threshold(blurred, thresh, 0, 255, Imgproc.THRESH_BINARY);
//            //----------END SkinMask processing------------
//
//            Mat res = new Mat();
//            float p = knn.findNearest(thresh, 1, res);
//            System.out.println(p + " " + res.dump()); // p or res.dump() = 1 -> 1 finger
//        }
    }
}

