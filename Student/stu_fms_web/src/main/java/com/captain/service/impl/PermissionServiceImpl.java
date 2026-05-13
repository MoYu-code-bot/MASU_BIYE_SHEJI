package com.captain.service.impl;

import com.captain.entity.po.Permission;
import com.captain.mapper.PermissionMapper;
import com.captain.service.PermissionService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.stereotype.Service;

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
public class PermissionServiceImpl extends ServiceImpl<PermissionMapper, Permission> implements PermissionService {
    @Override
    public List<Permission> getListByRoleId(Long roleId) {
        return baseMapper.getListByRoleId(roleId);
    }
}
