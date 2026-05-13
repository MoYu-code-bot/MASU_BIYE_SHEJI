package com.captain.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.captain.entity.po.Role;
import com.captain.entity.po.User;
import com.captain.entity.po.UserRole;
import com.captain.entity.vo.UserVo;
import com.captain.mapper.UserMapper;
import com.captain.service.RoleService;
import com.captain.service.UserRoleService;
import com.captain.service.UserService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

/**
 * <p>
 *  服务实现类
 * </p>
 *
 * @author lianhong
 * @since 2020-08-22
 */
@Service
public class UserServiceImpl extends ServiceImpl<UserMapper, User> implements UserService {

    @Autowired
    private RoleService roleService;
    @Autowired
    private UserRoleService userRoleService;

    @Override
    public User getByUsername(String username) {
        return baseMapper.selectOne(new QueryWrapper<User>()
                .eq("username",username));
    }

    @Override
    public Page getVoList(Page page, QueryWrapper<UserVo> wrapper) {
        List<UserVo> voList = baseMapper.getVoList(page,wrapper);
        return new Page().setRecords(voList);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public String registerStudentUser(User user) {
        String username = user.getUsername().trim();
        user.setUsername(username);
        if (getByUsername(username) != null) {
            return "用户名已存在";
        }
        if (user.getNickName() == null || user.getNickName().trim().isEmpty()) {
            user.setNickName(username);
        }
        if (!save(user)) {
            return "注册失败";
        }
        Role stuRole = roleService.getOne(new QueryWrapper<Role>().eq("role_mark", "student"));
        if (stuRole == null) {
            return "不存在学生角色，请联系管理员";
        }
        UserRole userRole = new UserRole();
        userRole.setUserId(user.getId());
        userRole.setRoleId(stuRole.getId());
        if (!userRoleService.save(userRole)) {
            return "分配默认角色失败";
        }
        return null;
    }
}
