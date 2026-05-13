package com.captain.service;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.captain.entity.po.User;
import com.baomidou.mybatisplus.extension.service.IService;
import com.captain.entity.vo.UserVo;

/**
 * <p>
 *  服务类
 * </p>
 *
 * @author lianhong
 * @since 2020-08-22
 */
public interface UserService extends IService<User> {

    User getByUsername(String username);

    Page getVoList(Page page,  QueryWrapper<UserVo> wrapper);

    /**
     * 自助注册：保存用户并绑定 student 角色。
     * @return 失败时的提示文案；成功返回 null
     */
    String registerStudentUser(User user);

}
