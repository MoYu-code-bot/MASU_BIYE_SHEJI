package com.captain.controller;

import com.captain.common.Result;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * 错误处理控制器
 */
@RestController
public class ErrorController {

    @GetMapping("/unauthorized")
    public Result unauthorized (){
        return Result.fail("您的操作未被授权");
    }

    @GetMapping("/unLogin")
    public Result unLogin (){
        return Result.fail("未登录，请登录");
    }
}
