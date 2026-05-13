package com.captain.config;

import com.captain.common.Result;
import org.apache.shiro.authz.UnauthenticatedException;
import org.apache.shiro.authz.UnauthorizedException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;

/**
 * 全局的异常处理方法
 * 只要程序里面出现了异常就会通过这里进行处理
 */
@RestControllerAdvice
public class GlobalExceptionHandler {
    private final Logger logger= LoggerFactory.getLogger(this.getClass());

    /**
     * 直接拦截所有异常
     * @param e
     * @return
     */
    @ExceptionHandler(value = Exception.class)
    public Result exceptionHandler(Exception e){
        Result result= Result.fail();
        result.setMsg(e.getMessage());
        //输出错误信息到控制台
//        logger.error(e.getMessage());
        e.printStackTrace();
        return result;
    }

    /**
     * 拦截由需要权限的请求发出来的异常
     * @return
     */
    @ExceptionHandler(value = UnauthenticatedException.class)
    public Result unauthenticatedException(){
        return Result.fail("未登录");
    }

    /**
     * 拦截由需要权限的请求发出来的未异常
     * @return
     */
    @ExceptionHandler(value = UnauthorizedException.class)
    public Result unauthorizedException(UnauthorizedException e){
        return Result.fail("未授权");
    }

}
