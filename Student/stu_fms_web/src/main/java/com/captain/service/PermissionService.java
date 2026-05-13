package com.captain.service;

import com.captain.entity.po.Permission;
import com.baomidou.mybatisplus.extension.service.IService;

import java.util.List;

/**
 * <p>
 *  服务类
 * </p>
 *
 * @author lianhong
 * @since 2020-08-22
 */
public interface PermissionService extends IService<Permission> {
    List<Permission> getListByRoleId(Long roleId);
}
