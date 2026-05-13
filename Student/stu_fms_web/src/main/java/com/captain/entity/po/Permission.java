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
public class Permission extends BaseEntity {

    private static final long serialVersionUID=1L;

    /**
     * 权限简称
     */
    private String title;

    /**
     * 权限描述
     */
    private String descr;

    /**
     * 权限访问路径
     */
    private String url;

    /**
     * 进行细细粒度划分时需要的一个标识
     */
    private String perms;

    /**
     * 父级权限ID
     */
    private Integer parentId;

    /**
     * 权限所属类型：菜单控制，按钮控制，页面控制
     */
    private String type;

    /**
     * 图标地址
     */
    private String icon;



}
