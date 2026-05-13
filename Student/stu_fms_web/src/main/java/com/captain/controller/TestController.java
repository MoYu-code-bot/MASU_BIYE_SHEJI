package com.captain.controller;

import com.captain.common.Result;
import org.apache.shiro.authz.annotation.RequiresPermissions;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class TestController {

    @RequiresPermissions("menu:archives-manage")
    @GetMapping("/hello")
    public Result hello(){
        return Result.success();
    }

}
