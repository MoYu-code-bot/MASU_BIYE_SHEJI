package com.captain.controller;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.captain.common.BaseRestController;
import com.captain.common.Result;
import com.captain.entity.po.Permission;
import com.captain.entity.po.RolePermission;
import com.captain.service.PermissionService;
import com.captain.service.RolePermissionService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.util.Assert;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/permission")
public class PermissionController extends BaseRestController<PermissionService,Permission> {
    @Autowired
    RolePermissionService rolePermissionService;

    @Override
    protected Result delete(Permission permission) {
        Assert.notNull(permission.getId(), "id不能为空");
        boolean res = baseService.removeById(permission.getId());
        if (!res) return Result.fail();
        //删除关联关系
        boolean permission_id = rolePermissionService.remove(new QueryWrapper<RolePermission>()
                .eq("permission_id", permission.getId()));
        if (!permission_id) return Result.fail();
        return Result.success();
    }

    @Override
    protected Result deleteBatch(List<Long> ids) {
        boolean res = baseService.removeByIds(ids);
        if (!res) return Result.fail();
        //删除关联关系
        boolean permission_id = rolePermissionService.remove(new QueryWrapper<RolePermission>()
                .in("permission_id", ids));
        if (!permission_id) return Result.fail();
        return Result.success();
    }

    /**
     * 获取所有的权限
     * @return
     */
    @GetMapping("/all")
    public Result getAll(){
        List<Permission> list = baseService.list();
        return Result.success(list);
    }

    @Override
    protected QueryWrapper<Permission> getSearchQueryWrapper(String searchValue, Permission permission) {
        QueryWrapper<Permission> wrapper = new QueryWrapper<>();
        if(StringUtils.isEmpty(searchValue)){
            wrapper.like("title",searchValue);
        }
        return wrapper;
    }
}
