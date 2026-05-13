package com.captain.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.captain.entity.po.Role;
import com.baomidou.mybatisplus.extension.service.IService;
import com.captain.entity.vo.RoleVo;

import java.util.List;

/**
 * <p>
 *  服务类
 * </p>
 *
 * @author lianhong
 * @since 2020-08-22
 */
public interface RoleService extends IService<Role> {

    List<RoleVo> getListByUid(Long uid);

    Page getList(Page page);

    boolean saveRoleVo(RoleVo roleVo);

    boolean updateRoleVo(RoleVo roleVo);

    boolean deleteRole(Long id);
}
