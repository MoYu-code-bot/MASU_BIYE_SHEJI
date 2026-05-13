package com.captain.service.impl;

import com.captain.common.Result;
import com.captain.entity.po.Audit;
import com.captain.entity.po.Dictionary;
import com.captain.mapper.AuditMapper;
import com.captain.service.AuditService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.captain.service.DictionaryService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.util.List;

/**
 * <p>
 *  服务实现类
 * </p>
 *
 * @author lxb
 * @since 2020-08-28
 */
@Service
public class
AuditServiceImpl extends ServiceImpl<AuditMapper, Audit> implements AuditService {
    @Autowired
    DictionaryService dictionaryService;

    @Override
    public Result passAudit(Long id) {
        //自己的待审核改成已经审核
        Audit audit = new Audit();
        audit.setId(id);
        audit.setIsAudited(1);
        int i = baseMapper.updateById(audit);
        if(i==0) return Result.fail("通过审核失败");
        //修改其他表的审核状态
        Audit newAudit = baseMapper.selectById(id);
        List<Dictionary> archiveTypes = dictionaryService.listByKey("archive_type");
        //找到对应的表名
        String tableName=null;
        for(Dictionary dictionary:archiveTypes){
            if(newAudit.getArchiveTypeCode().equals(dictionary.getDictCode())){
                tableName=dictionary.getDictValue();
                break;
            }
        }
        if(!StringUtils.isEmpty(tableName)){
            Integer res = baseMapper.updateOtherTableAuditStatus(tableName, newAudit.getArchiveId());
            if(res==0) return Result.fail("通过审核失败");
        }
        return Result.success("通过审核成功",null);
    }
}
