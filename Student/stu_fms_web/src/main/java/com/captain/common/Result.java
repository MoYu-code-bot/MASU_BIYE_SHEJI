package com.captain.common;

import lombok.Data;
import lombok.experimental.Accessors;

import java.io.Serializable;

@Data
@Accessors(chain=true)
public class Result implements Serializable {
    //操作状态：-1表示失败，0表示成功
    private Integer status;
    private String msg;
    private Object data;
    //携带的需要跳转到下一个地方的url
    private String action;

    public static Result success(){
        return success("success",null);
    }

    public static Result success(Object data){
        return success("success",data);
    }

    public static Result success(String msg, Object data){
            Result result=new Result();
            result.status=0;
            result.msg=msg;
            result.data=data;
            return result;
    }

    public static Result fail(){
        return fail("fail");
    }

    public static Result fail(String msg){
        Result result=new Result();
        result.status=-1;
        result.msg=msg;
        result.data=null;
        return result;
    }

}
