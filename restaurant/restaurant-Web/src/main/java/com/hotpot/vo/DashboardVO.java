package com.hotpot.vo;

import io.swagger.annotations.ApiModel;
import io.swagger.annotations.ApiModelProperty;
import lombok.Data;

import java.math.BigDecimal;
import java.util.List;
import java.util.Map;

@Data
@ApiModel(description = "首页数据概览")
public class DashboardVO {

    @ApiModelProperty("总门店数")
    private Long totalStores;

    @ApiModelProperty("总预订数")
    private Long totalReservations;

    @ApiModelProperty("总会员数")
    private Long totalCustomers;

    @ApiModelProperty("今日订单数")
    private Long todayOrderCount;

    @ApiModelProperty("今日营业额")
    private BigDecimal todayRevenue;

    @ApiModelProperty("总订单数")
    private Long totalOrderCount;

    @ApiModelProperty("总会员数(旧)")
    private Long totalMemberCount;

    @ApiModelProperty("近7天每日统计")
    private List<Map<String, Object>> dailyStats;
}
