package com.captain.controller;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.captain.common.BaseRestController;
import com.captain.common.Result;
import com.captain.entity.po.Role;
import com.captain.entity.vo.RoleVo;
import com.captain.service.RoleService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.util.Assert;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/role")
public class RoleController{
    @Autowired
    RoleService roleService;

    /**
     * 获取所有的角色数,分页
     * @return
     */
    @GetMapping("/all")
    public Result getAll(){
        List<Role> list = roleService.list();
        return Result.success(list);
    }

    /**
     * 获取某偶个用户的角色
     * @return
     */
    @GetMapping("/vo/getByUid")
    public Result getRoleByUid(Long uid){
        List<RoleVo> listByUid = roleService.getListByUid(uid);
        return Result.success(listByUid);
    }

    /**
     * 获取所有的角色数,分页
     * @return
     */
    @GetMapping("/list")
    public Result getList(Integer page,Integer size){
        Assert.notNull(page,"页数不能为空");
        Assert.notNull(size,"页大小不能为空");
        Page list = roleService.getList(new Page(page, size));
        return Result.success(list);
    }

    /**
     * 删除
     * @return
     */
    @PostMapping("/delete")
    public Result deleteById(@RequestBody  RoleVo roleVo){
        Assert.notNull(roleVo.getId(),"id不能为空");
        boolean b = roleService.deleteRole(roleVo.getId());
        if(b){
            return Result.success();
        }else{
            return Result.fail();
        }
    }

    /**
     * 新增一个角色和所属的权限
     * @param roleVo
     * @return
     */
    @PostMapping("/insert")
    public Result insert(@RequestBody  RoleVo roleVo){
        boolean b = roleService.saveRoleVo(roleVo);
        if(b){
            return Result.success();
        }else{
            return Result.fail();
        }
    }

    /**
     * 更新
     * @param roleVo
     * @return
     */
    @PostMapping("/update")
    public Result update(@RequestBody RoleVo roleVo){
        Assert.notNull(roleVo.getId(),"id不能为空");
        boolean b = roleService.updateRoleVo(roleVo);
        if(b){
            return Result.success();
        }else{
            return Result.fail();
        }
    }

}
