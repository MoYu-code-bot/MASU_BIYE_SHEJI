package com.captain.controller;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.captain.common.Result;
import com.captain.entity.po.Audit;
import com.captain.entity.vo.AuditVo;
import com.captain.service.ArchiveTypeService;
import com.captain.service.AuditService;
import com.captain.service.DictionaryService;
import com.captain.utils.ListToStringUtil;
import lombok.Data;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.util.Assert;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * 档案审核
 */
@RestController
@RequestMapping("/verify")
@Slf4j
public class ArchiveTypeController {

    @Autowired
    private ArchiveTypeService archiveTypeService;

    @Autowired
    private DictionaryService dictionaryService;

    @Autowired
    private AuditService auditService;



    @GetMapping("/list")
    public Result list(Integer page, Integer size) {
        Assert.notNull(page, "页数不能为空");
        Assert.notNull(size, "页大小不能为空");
        IPage<AuditVo> list = archiveTypeService.list(new Page(page, size));
        return Result.success(list);
    }

    @PostMapping("/delete")
    public Result delete(Integer id) {

        Assert.notNull(id, "id不能为空");
        boolean delete = archiveTypeService.delete(id);
        if (delete) {
            return Result.success();
        } else {
            return Result.fail();
        }
    }

    @PostMapping("/deleteBatch")
    public Result deleteBatch(@RequestBody IdList ids) {
        Assert.notNull(ids, "ids不能空");
        boolean b = archiveTypeService.myDeleteBatch(ListToStringUtil.convert(ids.getIds()));

        if (b) {
            return Result.success();
        } else {
            return Result.fail();
        }
    }

    //审核
    @GetMapping("/audit")
    public Result audit(Long id) {
        Assert.notNull(id, "id不能为空");
        return auditService.passAudit(id);
    }

    //提交审核
    @PostMapping("/submit")
    /**接口参数说明：
     * dictValue：dictionary里的dictValue，即档案类型：例如course，project
     * archiveId：档案类型id
     * stuId:学号
     */
    public Result submit(String dictValue,Long archiveId,String stuId) {


        Assert.notNull(dictValue,"档案类型为空");
        Assert.notNull(archiveId,"档案类型id为空");
        Assert.notNull(stuId,"学生id为空");

        Long dictCodeBydictValue = dictionaryService.getDictCodeBydictValue(dictValue);

        Audit audit = new Audit();
        audit.setArchiveTypeCode(dictCodeBydictValue);
        audit.setArchiveId(archiveId);
        audit.setStuId(stuId);
        audit.setIsAudited(0);

        boolean save = auditService.save(audit);

        if(save){
            return Result.success();
        }
        else{
            return Result.fail();
        }
    }


}

/**
 * ids的一个封装类
 * 解决application/json类型的数据传递
 */
@Data
class IdList {
    private List<Long> ids;
}
