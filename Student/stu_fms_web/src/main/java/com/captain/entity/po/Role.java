package com.captain.entity.po;

import com.captain.entity.po.BaseEntity;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * <p>
 *
 * </p>
 *
 * @author lianhong
 * @since 2020-08-22
 */
@Data
@EqualsAndHashCode(callSuper = true)
public class Role extends BaseEntity {

    private static final long serialVersionUID=1L;

    /**
     * 角色名字
     */
    private String roleName;

    /**
     * 角色描述
     */
    private String roleDesc;

    /**
     * 角色标识
     */
    private String roleMark;


}
