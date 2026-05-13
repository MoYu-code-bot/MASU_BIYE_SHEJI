package com.captain.mapper;

import com.captain.entity.po.Permission;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Select;
import org.springframework.stereotype.Repository;

import java.util.List;

/**
 * <p>
 *  Mapper 接口
 * </p>
 *
 * @author lianhong
 * @since 2020-08-22
 */
@Mapper
@Repository
public interface PermissionMapper extends BaseMapper<Permission> {

    @Select("select permission.* from role_permission\n" +
            "left join permission  on role_permission.permission_id=permission.id\n" +
            "where role_permission.role_id=#{roleId}")
    List<Permission> getListByRoleId(Long roleId);
}
