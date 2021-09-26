import org.oneflow.InferenceSession;
import org.oneflow.Option;
import org.oneflow.Tensor;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;


public class App {
    public static void main(String[] args) {
        System.out.println(Arrays.toString(args));
        String jobName = "mlp_inference";
        float[] image = readImage(args[0]);
        Option option = new Option();
        option.setDeviceTag(args[1]);
        option.setModelVersion("1");
        option.setControlPort(11245);
        option.setSavedModelDir(args[2]);

        Tensor imageTensor = Tensor.fromBlob(image, new long[]{ 1, 1, 28, 28 });
        Tensor tagTensor = Tensor.fromBlob(new int[]{ 1 }, new long[]{ 1 });
        Map<String, Tensor> tensorMap = new HashMap<>();
        tensorMap.put("image", imageTensor);
//        tensorMap.put("Input_15", tagTensor);

        InferenceSession inferenceSession = new InferenceSession(option);
        inferenceSession.open();

        long curTime = System.currentTimeMillis();
        for (int i = 0; i < Integer.parseInt(args[3]); i++) {
            Map<String, Tensor> resultMap = inferenceSession.run(jobName, tensorMap);
        }
//        for (Map.Entry<String, Tensor> entry : resultMap.entrySet()) {
//            Tensor resTensor = entry.getValue();
//            float[] resFloatArray = resTensor.getDataAsFloatArray();
//            System.out.println(Arrays.toString(resFloatArray));
//        }
        System.out.println(System.currentTimeMillis() - curTime);
        inferenceSession.close();
    }

    public static float[] readImage(String filePath) {
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
