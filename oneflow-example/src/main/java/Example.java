import org.oneflow.InferenceSession;
import org.oneflow.Option;
import org.oneflow.Tensor;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.util.HashMap;
import java.util.Map;

public class Example {

    public static void main(String[] args) {
        float[] image = readImage(args[0]);
        Tensor imageTensor = Tensor.fromBlob(image, new long[]{ 1, 1, 28, 28 });
        Tensor tagTensor = Tensor.fromBlob(new int[]{ 1 }, new long[]{ 1 });
        Map<String, Tensor> tensorMap = new HashMap<>();
        tensorMap.put("Input_14", imageTensor);
        tensorMap.put("Input_15", tagTensor);

        Option option = new Option();
        option.setDeviceTag("gpu")
                .setSavedModelDir("mnist_test/models")
                .setControlPort(12345)
                .setModelVersion("1");

        String jobName = "mlp_inference";
        InferenceSession inferenceSession = new InferenceSession(option);
        inferenceSession.open();
        Map<String, Tensor> resultMap = inferenceSession.run(jobName, tensorMap);
        for (Map.Entry<String, Tensor> entry : resultMap.entrySet()) {
            Tensor resTensor = entry.getValue();
            float[] resFloatArray = resTensor.getDataAsFloatArray();

            float maxVal = resFloatArray[0];
            int maxPos = 0;
            for (int i = 1; i < resFloatArray.length; i++) {
                if (maxVal < resFloatArray[i]) {
                    maxVal = resFloatArray[i];
                    maxPos = i;
                }
            }

            System.out.println("The prediction of image is: " + maxPos);
        }
        inferenceSession.close();
    }

    private static float[] readImage(String filePath) {
        File file = new File(filePath);
        BufferedImage image = null;
        try {
            image = ImageIO.read(file);
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        assert image != null;

        int width = image.getWidth();
        int height = image.getHeight();
        Raster raster = image.getRaster();
        float[] pixels = new float[width * height];
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                // Caution: transform the image
                pixels[i * width + j] = (raster.getSample(j, i, 0) - 128.0f) / 255.0f;
            }
        }
        return pixels;
    }
}
