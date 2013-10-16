package com.example.AndroidOpencvModule;

import android.app.Activity;
import android.content.Context;
import android.content.pm.ActivityInfo;
import android.graphics.Bitmap;
import android.graphics.drawable.BitmapDrawable;
import android.hardware.Camera;
import android.os.Bundle;
import android.util.Log;
import android.view.Window;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.LinearLayout;
import org.opencv.android.*;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.*;

public class MyActivity extends Activity
        implements CameraBridgeViewBase.CvCameraViewListener {


    private CameraBridgeViewBase openCvCameraView;
    private CascadeClassifier cascadeClassifier;
    private CascadeClassifier eyeCascadeClassifier;
    private Mat grayscaleImage;
    private int absoluteFaceSize;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    initializeOpenCVDependencies();
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    private File copyRawsToFilesystem(int rawFile) throws IOException {
        InputStream is = getResources().openRawResource(rawFile);
        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
        File mCascadeFile = new File(cascadeDir, "lbpcascade" + rawFile + ".xml");
        FileOutputStream os = new FileOutputStream(mCascadeFile);
        byte[] buffer = new byte[4096];
        int bytesRead;
        while ((bytesRead = is.read(buffer)) != -1) {
            os.write(buffer, 0, bytesRead);
        }
        is.close();
        os.close();

        return mCascadeFile;
    }

    private void initializeOpenCVDependencies() {
        try {
            // Copy the resource into a temp file so OpenCV can load it
            File fistFile = copyRawsToFilesystem(R.raw.fist);
            File eyeFile = copyRawsToFilesystem(R.raw.haarcascade_eye_tree_eyeglasses);
            // Load the cascade classifier
            cascadeClassifier = new CascadeClassifier(fistFile.getAbsolutePath());
            eyeCascadeClassifier = new CascadeClassifier(eyeFile.getAbsolutePath());
        } catch (Exception e) {
            Log.e("OpenCVActivity", "Error loading cascade", e);
        }

        // And we are ready to go
        openCvCameraView.enableView();
    }


    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);


        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);
        //setContentView(R.layout.main);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);


        openCvCameraView = new JavaCameraView(this, 1);
        //setContentView(openCvCameraView);
        setContentView(openCvCameraView);
        //((LinearLayout) findViewById(R.id.mainLayout)).addView(openCvCameraView);
        openCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        grayscaleImage = new Mat(height, width, CvType.CV_8UC4);


        // The faces will be a 5% of the height of the screen
        absoluteFaceSize = (int) (height * 0.1);
    }


    @Override
    public void onCameraViewStopped() {
    }

    private Mat findObjectsAndDrawThem(Mat aInputFrame, CascadeClassifier cascadeClassifier, Scalar color) {
        MatOfRect faces = new MatOfRect();
        /*aInputFrame = aInputFrame.t();*/

        // Use the classifier to detect faces
        if (cascadeClassifier != null) {
            cascadeClassifier.detectMultiScale(grayscaleImage, faces, 1.1, 2, 0,
                    new Size(absoluteFaceSize, absoluteFaceSize), new Size());
        }


        // If there are any faces found, draw a rectangle around it
        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++)
            Core.rectangle(aInputFrame, facesArray[i].tl(), facesArray[i].br(), color, 3);
        return aInputFrame;
    }

    @Override
    public Mat onCameraFrame(Mat aInputFrame) {
        // Create a grayscale image
        Mat rotatedMap = new Mat();
        Core.transpose(aInputFrame, rotatedMap);
        Core.flip(rotatedMap, rotatedMap, 0);

        Imgproc.cvtColor(rotatedMap, grayscaleImage, Imgproc.COLOR_RGBA2GRAY);
        //final Bitmap bitmap = Bitmap.createBitmap(rotatedMap.width(), rotatedMap.height(), Bitmap.Config.RGB_565);
        //Utils.matToBitmap(rotatedMap, bitmap);

        /*MyActivity.this.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                ((ImageView) findViewById(R.id.imageViewToChange)).setImageBitmap(bitmap);
            }
        });*/

        //findObjectsAndDrawThem(rotatedMap, cascadeClassifier, new Scalar(0, 255, 0, 255));
        findObjectsAndDrawThem(rotatedMap, eyeCascadeClassifier, new Scalar(0, 0, 255, 255));
        Core.flip(rotatedMap, rotatedMap, 1);
        Core.transpose(rotatedMap, aInputFrame);

        return aInputFrame;
    }


    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_6, this, mLoaderCallback);
    }
}
