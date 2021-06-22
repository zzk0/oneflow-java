package org.oneflow.example;

import org.oneflow.core.job.Env;

public class Example {

    public static void main(String[] args) {
        // example
        Env.EnvProto envProto = Env.EnvProto.newBuilder()
                .addMachine(Env.Machine.newBuilder().setId(0)
                        .setAddr("127.0.0.1"))
                .setCtrlPort(8888)
                .build();
        System.out.println(envProto.toString());
        if (org.oneflow.env.Env.isEnvInited()) {
            System.out.println("env inited");
        }
        else {
            System.out.println("try init");
            org.oneflow.env.Env.initEnv(envProto.toString());
        }

        // check it
        if (org.oneflow.env.Env.isEnvInited()) {
            System.out.println("ok");
        }
        else {
            System.out.println("no");
        }
    }

}
