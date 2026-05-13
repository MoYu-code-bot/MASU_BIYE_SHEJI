package com.captain.controller;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.captain.common.BaseRestController;
import com.captain.common.Result;
import com.captain.entity.po.Role;
import com.captain.entity.po.StuInfo;
import com.captain.entity.po.User;
import com.captain.entity.po.UserRole;
import com.captain.entity.vo.UserVo;
import com.captain.service.RoleService;
import com.captain.service.UserRoleService;
import com.captain.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.util.Assert;
import org.springframework.util.ResourceUtils;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.*;

/**
 * 用户相关的controller
 * @author lianhong
 */
@RestController
@RequestMapping("/user")
public class UserController extends BaseRestController<UserService,User> {
    @Autowired
    UserRoleService userRoleService;
    @Autowired
    RoleService roleService;

    @Override
    protected Result insert(User user) {
        //插入本身
        boolean res = baseService.save(user);
        if (!res) return Result.fail();
        //默认插入学生角色
        //获取学生这个角色
        Role stuRole = roleService.getOne(new QueryWrapper<Role>()
                .eq("role_mark", "student"));
        if(stuRole!=null){
            UserRole userRole = new UserRole();
            userRole.setUserId(user.getId());
            userRole.setRoleId(stuRole.getId());
            boolean save = userRoleService.save(userRole);
            if (!save) return Result.fail();
        }else{
            return Result.fail("不存在学生角色，作为默认的用户角色");
        }
        return Result.success();
    }

    @PostMapping("/vo/update")
    public Result updateUserVo(@RequestBody UserVo userVo){
        //更新
        boolean b = baseService.updateById(userVo);
        if(!b) return Result.fail();
        if(userVo.getRoleId()!=null){
            //判断有没有这个角色关系
            UserRole oldUserRole = userRoleService.getOne(new QueryWrapper<UserRole>()
                    .eq("user_id", userVo.getId()));
            if(oldUserRole==null){
                //添加关系
                UserRole userRole=new UserRole();
                userRole.setRoleId(userVo.getRoleId());
                userRole.setUserId(userVo.getId());
                boolean save = userRoleService.save(userRole);
                if(!save) return Result.fail();
            }else{
                //更新关系
                oldUserRole.setRoleId(userVo.getRoleId());
                boolean b1 = userRoleService.updateById(oldUserRole);
                if(!b1) return Result.fail();
            }
        }
        return Result.success();
    }


    /**
     * 搜索Vo
     * @param page
     * @param size
     * @param searchValue 需要模糊搜索的值
     * @param userVo
     * @return
     */
    @GetMapping("/vo/search")
    public Result fuzzySearch(Integer page, Integer size, String searchValue, UserVo userVo) {
        Assert.notNull(page, "页数不能为空");
        Assert.notNull(size, "页大小不能为空");
        QueryWrapper<UserVo> wrapper=new QueryWrapper();
        if(!StringUtils.isEmpty(searchValue)){
            //根据用户名和真名来模糊搜索
            wrapper.like("username",searchValue).or()
                    .like("real_name",searchValue).or()
                    .like("role_name",searchValue);
        }
        Page pageInfo = baseService.getVoList(new Page(page, size), wrapper);
        return Result.success(pageInfo);
    }

    @Override
    protected QueryWrapper<User> getSearchQueryWrapper(String searchValue, User user) {
        QueryWrapper<User> wrapper=new QueryWrapper();
        if(!StringUtils.isEmpty(searchValue)){
            //根据用户名和真名来模糊搜索
            wrapper.like("username",searchValue).or()
                    .like("real_name",searchValue);
            return wrapper;
        }
        return wrapper;
    }
}
