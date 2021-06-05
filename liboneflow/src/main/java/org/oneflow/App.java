package org.oneflow;

import org.oneflow.env.Env;
import org.oneflow.job.JobBuildAndInferCtx;

public class App 
{
    public static void main( String[] args )
    {
        if (Env.isEnvInited()) {
            System.out.println("env inited");
        }
        else {
            System.out.println("not init yet");
        }
    }
}