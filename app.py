
        color: var(--text);
    }

    .plan-item {
        margin-top: 8px;
        color: #314556;
        font-size: 14px;
        line-height: 1.7;
    }
.plan-item {
    margin-top: 8px;
    color: #314556;
    font-size: 14px;
    line-height: 1.7;
}

.phone-stage {
    display: grid;
    grid-template-columns: 1fr 80px 1fr;
    gap: 18px;
    align-items: center;
    margin-top: 22px;
}

.phone-arrow {
    text-align: center;
    font-size: 54px;
    font-weight: 800;
    color: rgba(217,154,37,0.65);
}

.phone-shell {
    max-width: 380px;
    margin: 0 auto;
    background: linear-gradient(180deg, #111111 0%, #1d1d1d 100%);
    border-radius: 42px;
    padding: 14px;
    box-shadow: 0 30px 70px rgba(15, 39, 64, 0.20);
}

.phone-screen {
    min-height: 760px;
    border-radius: 30px;
    overflow: hidden;
    background:
        radial-gradient(circle at top left, rgba(255,211,127,0.35), transparent 32%),
        linear-gradient(180deg, #fff9ef 0%, #fffdf8 100%);
    padding: 18px 18px 22px;
    position: relative;
}

.phone-dynamic {
    width: 118px;
    height: 24px;
    border-radius: 999px;
    background: #111111;
    margin: 0 auto 12px;
}

.phone-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 13px;
    color: #23394f;
    font-weight: 700;
    margin-bottom: 10px;
}

.phone-title {
    text-align: center;
    font-size: 24px;
    font-weight: 800;
    color: #1a2d41;
    margin: 8px 0 18px;
}

.segmented {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    padding: 6px;
    border-radius: 16px;
    background: rgba(35,57,79,0.08);
    margin-bottom: 18px;
}

.segmented span {
    text-align: center;
    padding: 10px 12px;
    border-radius: 12px;
    color: #5f7284;
    font-size: 13px;
    font-weight: 700;
}

.segmented .active {
    background: white;
    color: #1c3147;
    box-shadow: 0 10px 20px rgba(15,39,64,0.08);
