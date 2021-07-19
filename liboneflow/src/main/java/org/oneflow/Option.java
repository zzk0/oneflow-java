package org.oneflow;

public class Option {

    /**
     * The tag of device can be: "gpu" or "cpu"
     */
    private String deviceTag;

    /**
     * The number of devices used, the default value is 1
     */
    private int deviceNum;

    /**
     * The setting of distributed training: mirrored or consistent
     */
    private boolean mirroredView;

    /**
     * Control port: default 11235
     */
    private int controlPort;

    public Option() {
        this.deviceTag = "gpu";
        this.deviceNum = 1;
        this.mirroredView = false;
    }

    public Option(String deviceTag, int deviceNum, boolean mirroredView) {
        this.deviceTag = deviceTag;
        this.deviceNum = deviceNum;
        this.mirroredView = mirroredView;
    }

    public String getDeviceTag() {
        return deviceTag;
    }

    public void setDeviceTag(String deviceTag) {
        this.deviceTag = deviceTag;
    }

    public int getDeviceNum() {
        return deviceNum;
    }

    public void setDeviceNum(int deviceNum) {
        this.deviceNum = deviceNum;
    }

    public boolean isMirroredView() {
        return mirroredView;
    }

    public void setMirroredView(boolean mirroredView) {
        this.mirroredView = mirroredView;
    }

}
