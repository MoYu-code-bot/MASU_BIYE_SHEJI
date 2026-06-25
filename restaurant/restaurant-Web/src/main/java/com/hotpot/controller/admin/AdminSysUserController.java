package com.hotpot.controller.admin;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.hotpot.common.BusinessException;
import com.hotpot.common.PageResult;
import com.hotpot.common.Result;
import com.hotpot.dto.PageQuery;
import com.hotpot.entity.SysUser;
import com.hotpot.service.SysUserService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiOperation;
import lombok.RequiredArgsConstructor;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.*;

import java.util.Collection;

@Api(tags = "B端-系统用户管理")
@RestController
@RequestMapping("/admin/users")
@RequiredArgsConstructor
public class AdminSysUserController {

    private final SysUserService sysUserService;
    private final PasswordEncoder passwordEncoder;

    /**
     * 获取当前登录用户的角色
     */
    private String getCurrentRole() {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth != null && auth.getAuthorities() != null) {
            for (GrantedAuthority ga : auth.getAuthorities()) {
                String authority = ga.getAuthority();
                if (authority.startsWith("ROLE_")) {
                    return authority.substring(5); // 去掉 ROLE_ 前缀
                }
            }
        }
        return "STAFF";
    }

    @ApiOperation("分页查询用户（MANAGER不可见ADMIN）")
    @GetMapping
    @PreAuthorize("hasAnyRole('ADMIN','MANAGER')")
    public Result<PageResult<SysUser>> page(PageQuery query) {
        String role = getCurrentRole();
        Page<SysUser> page = new Page<>(query.getPageNum(), query.getPageSize());
        LambdaQueryWrapper<SysUser> wrapper = new LambdaQueryWrapper<>();
        // 店长不能看到ADMIN用户
        if ("MANAGER".equals(role)) {
            wrapper.ne(SysUser::getRole, "ADMIN");
        }
        wrapper.orderByDesc(SysUser::getCreateTime);
        return Result.success(PageResult.of(sysUserService.page(page, wrapper)));
    }

    @ApiOperation("新增用户（仅ADMIN可新增任意角色，MANAGER只能新增STAFF）")
    @PostMapping
    @PreAuthorize("hasAnyRole('ADMIN','MANAGER')")
    public Result<?> create(@RequestBody SysUser sysUser) {
        String currentRole = getCurrentRole();
        // 店长不能创建ADMIN或MANAGER
        if ("MANAGER".equals(currentRole) && !"STAFF".equals(sysUser.getRole())) {
            throw new BusinessException("店长只能新增员工账号");
        }
        // 检查用户名唯一性
        SysUser exist = sysUserService.getOne(new LambdaQueryWrapper<SysUser>()
                .eq(SysUser::getUsername, sysUser.getUsername()));
        if (exist != null) {
            throw new BusinessException("用户名已存在");
        }
        sysUser.setPassword(passwordEncoder.encode(sysUser.getPassword()));
        sysUser.setStatus(1);
        sysUserService.save(sysUser);
        return Result.success("创建成功");
    }

    @ApiOperation("修改用户信息（不可修改ADMIN角色，不可降级其他ADMIN）")
    @PutMapping("/{userId}")
    @PreAuthorize("hasAnyRole('ADMIN','MANAGER')")
    public Result<?> update(@PathVariable Long userId, @RequestBody SysUser updateInfo) {
        String currentRole = getCurrentRole();
        SysUser target = sysUserService.getById(userId);
        if (target == null) {
            throw new BusinessException("用户不存在");
        }
        // 店长不能操作ADMIN用户
        if ("MANAGER".equals(currentRole) && "ADMIN".equals(target.getRole())) {
            throw new BusinessException("店长无权操作管理员账号");
        }
        // 店长不能把用户改为ADMIN或MANAGER
        if ("MANAGER".equals(currentRole) && updateInfo.getRole() != null
                && ("ADMIN".equals(updateInfo.getRole()) || "MANAGER".equals(updateInfo.getRole()))) {
            throw new BusinessException("店长只能设置员工角色");
        }
        // 不能禁用或修改最后一个ADMIN
        if ("ADMIN".equals(target.getRole())) {
            long adminCount = sysUserService.count(new LambdaQueryWrapper<SysUser>()
                    .eq(SysUser::getRole, "ADMIN")
                    .eq(SysUser::getStatus, 1));
            if (adminCount <= 1 && updateInfo.getStatus() != null && updateInfo.getStatus() != 1) {
                throw new BusinessException("不能禁用唯一的管理员账号");
            }
        }
        updateInfo.setId(userId);
        updateInfo.setPassword(null); // 不允许通过此接口改密码
        sysUserService.updateById(updateInfo);
        return Result.success("修改成功");
    }

    @ApiOperation("删除用户（不可删除ADMIN）")
    @DeleteMapping("/{userId}")
    @PreAuthorize("hasRole('ADMIN')")
    public Result<?> delete(@PathVariable Long userId) {
        SysUser target = sysUserService.getById(userId);
        if (target == null) {
            throw new BusinessException("用户不存在");
        }
        if ("ADMIN".equals(target.getRole())) {
            throw new BusinessException("不能删除管理员账号");
        }
        sysUserService.removeById(userId);
        return Result.success("删除成功");
    }

    @ApiOperation("修改用户状态（启用/禁用）")
    @PutMapping("/updateStatus")
    @PreAuthorize("hasAnyRole('ADMIN','MANAGER')")
    public Result<?> updateStatus(@RequestParam Long userId, @RequestParam Integer status) {
        String currentRole = getCurrentRole();
        SysUser target = sysUserService.getById(userId);
        if (target == null) {
            throw new BusinessException("用户不存在");
        }
        if ("MANAGER".equals(currentRole) && "ADMIN".equals(target.getRole())) {
            throw new BusinessException("店长无权操作管理员账号");
        }
        SysUser user = new SysUser();
        user.setId(userId);
        user.setStatus(status);
        sysUserService.updateById(user);
        return Result.success();
    }
}
