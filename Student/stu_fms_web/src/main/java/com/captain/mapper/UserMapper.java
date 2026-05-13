package com.captain.mapper;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.captain.entity.po.User;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.captain.entity.vo.UserVo;
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
public interface UserMapper extends BaseMapper<User> {

    @Select("select user.*,role.id as role_id,role.role_name from user\n" +
            "left join user_role on user.id=user_role.user_id\n" +
            "left join role ON user_role.role_id=role.id ${ew.customSqlSegment}")
    List<UserVo> getVoList(Page page, @Param("ew") QueryWrapper<UserVo> wrapper);
}
