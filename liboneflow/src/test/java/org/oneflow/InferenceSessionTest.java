package org.oneflow;

import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.*;

public class InferenceSessionTest {

//    @Test
//    public void testInference() {
//        InferenceSession session = new InferenceSession();
//        session.open();
//        session.loadSavedModel("./models");
//        session.launch();
//
//        // input
//        float[] image = readImage("./7.png");
//        Tensor tagTensor = Tensor.fromBlob(new int[]{ 0 }, new long[]{ 1 }, DType.INT);
//        Tensor imageTensor = Tensor.fromBlob(image, new long[]{ 28, 28 }, DType.FLOAT);
//        Map<String, Tensor> tensors = new HashMap<>();
//        tensors.put("Input_14", imageTensor);
//        tensors.put("Input_15", tagTensor);
//
//        // forward
//        Tensor[] result = session.run("mlp_inference", tensors);
//        Tensor prediction = result[0];
//
//        // close
//        session.close();
//
//        // assert
//        float[] vector = prediction.getFloatData();
//        assertEquals(10, vector.length);
//        float[] expectedVector = { -129.57167f, -89.084816f, -139.21355f , -103.455025f, -9.179366f,
//                -69.568474f, -133.39594f,  -16.204329f, -114.90876f,  -47.933548f };
//        float delta = 0.0001f;
//        for (int i = 0; i < 10; i++) {
//            assertEquals(expectedVector[i], vector[i], delta);
//        }
//    }
//
//    public static float[] readImage(String filePath) {
//        File file = new File(filePath);
//        BufferedImage image = null;
//        try {
//            image = ImageIO.read(file);
//        }
//        catch (Exception e) {
//            e.printStackTrace();
//        }
//        assert image != null;
//
//        int width = image.getWidth();
//        int height = image.getHeight();
//        Raster raster = image.getRaster();
//        float[] pixels = new float[width * height];
//        for (int i = 0; i < width; i++) {
//            for (int j = 0; j < height; j++) {
//                // Caution: transform the image
//                pixels[i * width + j] = (raster.getSample(j, i, 0) - 128.0f) / 255.0f;
//            }
//        }
//        return pixels;
//    }

}
