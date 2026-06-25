package com.hotpot.service;

import com.baomidou.mybatisplus.extension.service.IService;
import com.hotpot.common.PageResult;
import com.hotpot.dto.LoginRequest;
import com.hotpot.dto.PageQuery;
import com.hotpot.dto.PasswordUpdateRequest;
import com.hotpot.entity.SysUser;
import com.hotpot.vo.LoginVO;

public interface SysUserService extends IService<SysUser> {

    LoginVO login(LoginRequest request);

    SysUser getByUsername(String username);

    void updatePassword(Long userId, PasswordUpdateRequest request);

    SysUser getUserInfo(Long userId);

    PageResult<SysUser> pageQuery(PageQuery query);
}
