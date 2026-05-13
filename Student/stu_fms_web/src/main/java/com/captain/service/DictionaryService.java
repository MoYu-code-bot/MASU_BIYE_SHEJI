package com.captain.service;

import com.captain.entity.po.Dictionary;
import com.baomidou.mybatisplus.extension.service.IService;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/**
 * <p>
 *  服务类
 * </p>
 *
 * @author lianhong
 * @since 2020-08-22
 */
public interface DictionaryService extends IService<Dictionary> {

    Long getDictCodeBydictValue(String name);

    List<Dictionary> listByKey(String key);

    Dictionary getOneByKeyAndCode(String key,Long code);
}
