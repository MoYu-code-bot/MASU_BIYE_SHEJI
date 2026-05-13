package com.captain.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.captain.entity.po.PhysicalArchives;
import com.captain.entity.po.PhysicalBorrRecord;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

/**
 * <p>
 *  Mapper 接口
 * </p>
 *
 * @author lxb
 * @since 2020-08-29
 */
public interface PhysicalBorrRecordMapper extends BaseMapper<PhysicalBorrRecord> {

    //通过用户id查询借阅记录
    @Select("SELECT * FROM physical_borr_record  WHERE  username_id= #{usernameId}")
    List<PhysicalBorrRecord> findOneByusernameID(@Param("usernameId") Long usernameId);
}
