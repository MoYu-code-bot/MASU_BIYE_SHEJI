package com.hotpot.controller.api;

import com.hotpot.common.Result;
import com.hotpot.dto.MemberLoginRequest;
import com.hotpot.dto.MemberRegisterRequest;
import com.hotpot.service.CustomerService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import io.swagger.annotations.ApiParam;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import javax.validation.Valid;

@Api(tags = "C端-认证接口")
@RestController
@RequestMapping("/api/auth")
@RequiredArgsConstructor
public class AuthApiController {

    private final CustomerService customerService;

    @ApiOperation("会员登录")
    @PostMapping("/login")
    public Result<String> login(@ApiParam("登录请求") @Valid @RequestBody MemberLoginRequest request) {
        String token = customerService.login(request.getPhone(), request.getPassword());
        return Result.success(token);
    }

    @ApiOperation("会员注册")
    @PostMapping("/register")
    public Result<String> register(@ApiParam("注册请求") @Valid @RequestBody MemberRegisterRequest request) {
        String token = customerService.register(request.getPhone(), request.getPassword(), request.getNickname());
        return Result.success(token);
    }
}
