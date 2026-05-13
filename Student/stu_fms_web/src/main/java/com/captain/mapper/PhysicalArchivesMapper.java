package com.captain.mapper;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.captain.entity.po.PhysicalArchives;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

/**
 * <p>
 *  Mapper 接口
 * </p>
 *
 * @author lxb
 * @since 2020-08-28
 */
public interface PhysicalArchivesMapper extends BaseMapper<PhysicalArchives> {

    //出库
    @Update("UPDATE physical_archives SET audit_status_code = '6' WHERE id = #{id}")
    boolean outStock(@Param("id") Long id);

    //查询所有入库的档案
    @Select("SELECT * FROM physical_archives  WHERE  audit_status_code = '4' ")
    IPage<PhysicalArchives> findListByUsernameID(Page<PhysicalArchives> page);



}
