package com.captain.entity.vo;

import com.captain.entity.po.Permission;
import com.captain.entity.po.Role;
import lombok.Data;

import java.util.List;

/**
 * 角色VO
 */
@Data
public class RoleVo  extends Role {
    /**
     * 该角色拥有的权限
     */
    List<Permission> permissions;
}
