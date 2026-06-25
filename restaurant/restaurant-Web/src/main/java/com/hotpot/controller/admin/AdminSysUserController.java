package com.hotpot.controller.admin;

import com.hotpot.common.PageResult;
import com.hotpot.common.Result;
import com.hotpot.dto.PageQuery;
import com.hotpot.entity.SysUser;
import com.hotpot.service.SysUserService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiImplicitParam;
import io.swagger.annotations.ApiOperation;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@Api(tags = "B端-系统用户管理")
@RestController
@RequestMapping("/admin/users")
@RequiredArgsConstructor
public class AdminSysUserController {

    private final SysUserService sysUserService;

    @ApiOperation("分页查询用户")
    @GetMapping
    public Result<PageResult<SysUser>> page(PageQuery query) {
        return Result.success(sysUserService.pageQuery(query));
    }

    @ApiOperation("修改用户状态")
    @PutMapping("/updateStatus")
    public Result<?> updateStatus(@RequestParam Long userId, @RequestParam Integer status) {
        SysUser user = new SysUser();
        user.setId(userId);
        user.setStatus(status);
        sysUserService.updateById(user);
        return Result.success();
    }
}
