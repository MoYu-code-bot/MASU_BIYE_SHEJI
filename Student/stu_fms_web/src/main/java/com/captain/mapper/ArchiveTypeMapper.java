package com.captain.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.captain.entity.vo.AuditVo;
import org.apache.ibatis.annotations.*;

import java.util.List;

@Mapper
public interface ArchiveTypeMapper extends BaseMapper<AuditVo>{

    @Select("SELECT  a.id,a.archive_id,a.stu_id,a.is_audited,d.dict_value,d.dict_desc,d.dict_code FROM audit a LEFT JOIN dictionary d ON a.archive_type_code = d.dict_code AND d.dict_key ='archive_type'")
    IPage<AuditVo> list(Page page);

    @Update("UPDATE audit SET is_audited = 1 WHERE id = #{id}")
    boolean audit(@Param("id") Long id);

    @Delete("delete from audit where id in (#{ids})")
    boolean myDeleteBatch(@Param("ids") String ids);
}
