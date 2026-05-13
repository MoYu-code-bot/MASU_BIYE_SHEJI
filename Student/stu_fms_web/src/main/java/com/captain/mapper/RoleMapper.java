package com.captain.mapper;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.captain.entity.po.Role;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.captain.entity.vo.RoleVo;
import org.apache.ibatis.annotations.*;

import java.util.List;

/**
 * <p>
 *  Mapper 接口
 * </p>
 *
 * @author lianhong
 * @since 2020-08-22
 */
public interface RoleMapper extends BaseMapper<Role> {

    @Select("select role.* from role\n" +
            "${ew.customSqlSegment}")
    @Results({
            @Result(column = "id",property = "id"),
            @Result(column = "role_name",property = "roleName"),
            @Result(column = "role_desc",property = "roleDesc"),
            @Result(column = "role_mark",property = "roleMark"),
            @Result(column = "id",property = "permissions"
                    ,many = @Many(select = "com.captain.mapper.PermissionMapper.getListByRoleId")),
    })
    List<RoleVo> getList(Page page,@Param("ew") QueryWrapper queryWrapper);

    @Select("select role.* from user_role\n" +
            "left join role  on user_role.role_id=role.id\n" +
            "where user_role.user_id=#{uid}")
    @Results({
            @Result(column = "id",property = "id"),
            @Result(column = "id",property = "permissions"
            ,many = @Many(select = "com.captain.mapper.PermissionMapper.getListByRoleId")),
    })
    List<RoleVo> getListByUid(Long uid);
}
