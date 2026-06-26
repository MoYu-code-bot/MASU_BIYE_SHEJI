package com.hotpot.controller.admin;

import com.hotpot.common.Result;
import com.hotpot.dto.LoginRequest;
import com.hotpot.service.SysUserService;
import com.hotpot.vo.LoginVO;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import io.swagger.annotations.ApiParam;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import javax.validation.Valid;

@Api(tags = "B端-认证管理")
@RestController
@RequestMapping("/admin/auth")
@RequiredArgsConstructor
public class AdminAuthController {

    private final SysUserService sysUserService;

    @ApiOperation("管理员登录")
    @PostMapping("/login")
    public Result<LoginVO> login(@ApiParam("登录请求") @Valid @RequestBody LoginRequest request) {
        LoginVO vo = sysUserService.login(request);
        return Result.success(vo);
    }
}
