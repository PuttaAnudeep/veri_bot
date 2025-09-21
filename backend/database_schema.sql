-- Background Check System Database Schema

CREATE TABLE Subject (
    subject_id INT PRIMARY KEY,
    subject_name NVARCHAR(100),
    subject_alias NVARCHAR(50),
    subject_contact VARCHAR(40),
    subject_address1 NVARCHAR(200),
    subject_address2 NVARCHAR(200),
    sbj_city NVARCHAR(100)
);

CREATE TABLE Company (
    comp_id INT PRIMARY KEY,
    comp_name NVARCHAR(200),
    comp_code NVARCHAR(20)
);

CREATE TABLE package (
    package_code INT PRIMARY KEY,
    package_name VARCHAR(150),
    package_price MONEY,
    comp_code NVARCHAR(20)
);

CREATE TABLE search_type (
    search_type_code NVARCHAR(20) PRIMARY KEY,
    search_type VARCHAR(50),
    search_type_category VARCHAR(50)
);

CREATE TABLE order_request (
    order_id INT PRIMARY KEY,
    order_package_id NVARCHAR(40),
    order_subject_id INT,
    order_company_code NVARCHAR(20),
    order_status VARCHAR(20),
    order_packcage_code INT
);

CREATE TABLE search (
    search_id INT PRIMARY KEY,
    package_req_id NVARCHAR(20),
    subject_id INT,
    search_type_code NVARCHAR(20),
    search_status NVARCHAR(12),
    county_name NVARCHAR(50),
    state_code NVARCHAR(5),
    pkg_code INT,
    sub_status NVARCHAR(200)
);

CREATE TABLE search_status (
    status_code NVARCHAR(12) PRIMARY KEY,
    status NVARCHAR(100)
);

CREATE TABLE order_status (
    status_code NVARCHAR(12) PRIMARY KEY,
    status_description NVARCHAR(100)
);

-- Key Relationships (based on your schema):
-- Search.subject_id -> Subject.subject_id
-- Search.search_type_code -> Search_Type.search_type_code
-- Search.search_status -> Search_status.status_code
-- Order_Request.order_subject_id -> Subject.subject_id
-- Order_Request.order_status -> Order_status.status_code
-- Order_Request.order_company_code -> Company.comp_code
-- Package.comp_code -> Company.comp_code