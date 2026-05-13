package com.captain.mapper;

import com.captain.entity.po.Dictionary;
import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

/**
 * <p>
 *  Mapper 接口
 * </p>
 *
 * @author lianhong
 * @since 2020-08-22
 */
public interface DictionaryMapper extends BaseMapper<Dictionary> {

    @Select("SELECT dict_code FROM dictionary WHERE dict_value = #{name}")
    Long getDictCodeBydictValue(@Param("name") String name);
}
